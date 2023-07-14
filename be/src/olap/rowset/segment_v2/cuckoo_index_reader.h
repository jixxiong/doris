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
#include "io/fs/file_reader_writer_fwd.h"
#include "olap/itoken_extractor.h"
#include "olap/rowset/segment_v2/bloom_filter.h"
#include "olap/rowset/segment_v2/fixed_cuckoo_table.h"
#include "olap/rowset/segment_v2/indexed_column_reader.h"
#include "olap/rowset/segment_v2/indexed_column_writer.h"
#include "olap/types.h"
#include "olap/utils.h"
#include "util/slice.h"
#include "vec/common/arena.h"
#include "vec/data_types/data_type.h"
#include "vec/data_types/data_type_factory.hpp"

namespace doris {

namespace segment_v2 {

class CuckooIndexIterator;

class CuckooIndexReader {
public:
    CuckooIndexReader(io::FileReaderSPtr file_reader, const CuckooIndexPB* cuckoo_index_meta)
            : _file_reader(file_reader),
              _type_info(get_scalar_type_info<FieldType::OLAP_FIELD_TYPE_VARCHAR>()),
              _cuckoo_index_meta(cuckoo_index_meta) {}

    Status load(bool use_page_cache, bool kept_in_memory) {
        const IndexedColumnMetaPB& cuckoo_index_meta = _cuckoo_index_meta->cuckoo_index();

        _cuckoo_index_reader =
                std::make_unique<IndexedColumnReader>(_file_reader, cuckoo_index_meta);
        RETURN_IF_ERROR(_cuckoo_index_reader->load(use_page_cache, kept_in_memory));
        return Status::OK();
    }

    Status read_cuckoo_table(std::unique_ptr<CuckooTable>* table) {
        OlapStopWatch watch;
        auto num_layers = _cuckoo_index_meta->num_layers();
        auto data_type =
                vectorized::DataTypeFactory::instance().create_data_type(type_info()->type(), 1, 0);
        auto column = data_type->create_column();
        auto cuckoo_index_iter = IndexedColumnIterator(_cuckoo_index_reader.get());
        RETURN_IF_ERROR(cuckoo_index_iter.seek_to_ordinal(0));
        size_t num_read = num_layers;
        RETURN_IF_ERROR(cuckoo_index_iter.next_batch(&num_read, column));
        DCHECK(num_layers == num_read);
        size_t len_table = column->get_data_at(0).size / sizeof(CuckooTable::Slot);

        *table = std::make_unique<CuckooTable>();
        CuckooTableOptions opts;
        opts.layer_size = len_table;
        opts.read_only = true;
        RETURN_IF_ERROR((*table)->init(opts));

        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            auto value = column->get_data_at(layer).to_slice();
            RETURN_IF_ERROR((*table)->set_layer(layer, value));
        }
        CuckooTableStat::get_stat()->load_watch += watch.get_elapse_time_us();
        return Status::OK();
    }

    [[nodiscard]] const TypeInfo* type_info() const { return _type_info; }

private:
    io::FileReaderSPtr _file_reader;
    const TypeInfo* _type_info;
    const CuckooIndexPB* _cuckoo_index_meta;
    std::unique_ptr<IndexedColumnReader> _cuckoo_index_reader;
};

} // namespace segment_v2

} // namespace doris
