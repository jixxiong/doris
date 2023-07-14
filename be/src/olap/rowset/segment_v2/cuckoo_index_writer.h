// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <gen_cpp/segment_v2.pb.h>

#include "common/status.h"
#include "io/fs/file_writer.h"
#include "olap/itoken_extractor.h"
#include "olap/rowset/segment_v2/bloom_filter.h"
#include "olap/rowset/segment_v2/fixed_cuckoo_table.h"
#include "olap/rowset/segment_v2/indexed_column_writer.h"
#include "olap/types.h"
#include "util/slice.h"
#include "vec/common/arena.h"

namespace doris {

namespace segment_v2 {

class CuckooIndexWriter {
public:
    CuckooIndexWriter() = default;
    ~CuckooIndexWriter() = default;

    DISALLOW_COPY_AND_ASSIGN(CuckooIndexWriter);

    Status init(CuckooTableOptions opts) {
        _table = std::make_unique<CuckooTable>();
        _opts = opts;
        _opts.layer_size = 0;
        _table->init(opts);
        _buffer = std::make_unique<std::vector<CuckooTable::Slot>>();
        return Status::OK();
    }

    Status add_item(const Slice& key) {
        // this function assume that keys are added in order
        _buffer->emplace_back(CuckooHashCodeGenerator::get_finger(key), _num_rows);
        _num_rows += 1;
        return Status::OK();
    }

    [[nodiscard]] uint32_t num_rows() const { return _num_rows; }

    Status finish(io::FileWriter* file_writer, ColumnIndexMetaPB* index_meta) {
        RETURN_IF_ERROR(finalize_cuckoo_table());

        index_meta->set_type(CUCKOO_INDEX);
        CuckooIndexPB* meta = index_meta->mutable_cuckoo_index();
        meta->set_num_layers(_table->layers());

        const auto* cuckoo_type_info = get_scalar_type_info<FieldType::OLAP_FIELD_TYPE_VARCHAR>();
        IndexedColumnWriterOptions options;
        options.write_ordinal_index = true;
        options.write_value_index = false;
        options.encoding = PLAIN_ENCODING;
        IndexedColumnWriter cuckoo_index_writer(options, cuckoo_type_info, file_writer);
        RETURN_IF_ERROR(cuckoo_index_writer.init());
        for (uint64_t layer = 0, offset = 0; layer < _table->layers(); ++layer) {
            Slice data(reinterpret_cast<uint8_t*>(_table->_table.data()) + offset,
                       _table->_len_table * sizeof(CuckooTable::Slot));
            offset += _table->_len_table * sizeof(CuckooTable::Slot);
            cuckoo_index_writer.add(&data);
        }
        RETURN_IF_ERROR(cuckoo_index_writer.finish(meta->mutable_cuckoo_index()));
        _size = _table->size();
        return Status::OK();
    }

    [[nodiscard]] uint64_t size() const { return _size; }

private:
    Status finalize_cuckoo_table() {
        _opts.do_shrink_to_fit = false;
        _opts.layer_size =
                std::max(64ul, static_cast<size_t>(ceil(_num_rows / 3 / _opts.max_density)));
        RETURN_IF_ERROR(_table->init(_opts));

        for (auto slot : *_buffer) {
            RETURN_IF_ERROR(_table->_insert(slot.finger(), slot.value()));
        }
        return Status::OK();
    }

    std::unique_ptr<CuckooTable> _table;
    std::unique_ptr<std::vector<CuckooTable::Slot>> _buffer;
    CuckooTableOptions _opts;
    uint32_t _num_rows {0};
    uint64_t _size {0};
};

} // namespace segment_v2

} // namespace doris
