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

#include <glog/logging.h>
#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "common/status.h"
#include "io/fs/file_reader_writer_fwd.h"
#include "olap/rowset/segment_v2/bloom_filter.h"
#include "olap/rowset/segment_v2/bloom_filter_index_writer.h"
#include "olap/rowset/segment_v2/cuckoo_index_reader.h"
#include "olap/rowset/segment_v2/cuckoo_index_writer.h"
#include "olap/rowset/segment_v2/indexed_column_reader.h"
#include "olap/rowset/segment_v2/indexed_column_writer.h"
#include "util/faststring.h"
#include "util/slice.h"

namespace doris {
class TypeInfo;

namespace io {
class FileWriter;
} // namespace io
namespace segment_v2 {
class PrimaryKeyIndexMetaPB;
} // namespace segment_v2

class HashPrimaryKeyIndexIterator;

// Build index for primary key.
// The primary key index is designed in a similar way like RocksDB
// Partitioned Index, which is created in the segment file when MemTable flushes.
// Index is stored in multiple pages to leverage the IndexedColumnWriter.
//
// NOTE: for now, it's only used when unique key merge-on-write property enabled.
class PrimaryKeyIndexBuilder {
public:
    PrimaryKeyIndexBuilder(io::FileWriter* file_writer, size_t seq_col_length)
            : _file_writer(file_writer),
              _num_rows(0),
              _size(0),
              _disk_size(0),
              _seq_col_length(seq_col_length) {}

    Status init();

    Status add_item(const Slice& key);

    [[nodiscard]] uint32_t num_rows() const { return _num_rows; }

    [[nodiscard]] uint64_t size() const { return _size; }

    uint64_t disk_size() const { return _disk_size; }

    Slice min_key() { return Slice(_min_key.data(), _min_key.size() - _seq_col_length); }
    Slice max_key() { return Slice(_max_key.data(), _max_key.size() - _seq_col_length); }

    Status finalize(segment_v2::PrimaryKeyIndexMetaPB* meta);

private:
    io::FileWriter* _file_writer = nullptr;
    uint32_t _num_rows;
    uint64_t _size;
    uint64_t _disk_size;
    size_t _seq_col_length;

    faststring _min_key;
    faststring _max_key;
    std::unique_ptr<segment_v2::IndexedColumnWriter> _primary_key_index_builder;
    std::unique_ptr<segment_v2::CuckooIndexWriter> _cuckoo_index_builder;
    std::unique_ptr<segment_v2::BloomFilterIndexWriter> _bloom_filter_index_builder;
};

class PrimaryKeyIndexReader {
public:
    PrimaryKeyIndexReader() : _index_parsed(false), _bf_parsed(false) {}

    Status parse_index(io::FileReaderSPtr file_reader,
                       const segment_v2::PrimaryKeyIndexMetaPB& meta);

    Status parse_bf(io::FileReaderSPtr file_reader, const segment_v2::PrimaryKeyIndexMetaPB& meta);

    Status new_iterator(std::unique_ptr<segment_v2::IndexedColumnIterator>* index_iterator) const {
        DCHECK(_index_parsed);
        index_iterator->reset(new segment_v2::IndexedColumnIterator(_index_reader.get()));
        return Status::OK();
    }

    [[nodiscard]] HashPrimaryKeyIndexIterator new_hash_iterator() const;

    [[nodiscard]] const TypeInfo* type_info() const {
        DCHECK(_index_parsed);
        return _index_reader->type_info();
    }

    // verify whether exist in BloomFilter
    bool check_present(const Slice& key) {
        DCHECK(_bf_parsed);
        return _bf->test_bytes(key.data, key.size);
    }

    [[nodiscard]] uint32_t num_rows() const {
        DCHECK(_index_parsed);
        return _index_reader->num_values();
    }

    uint64_t get_bf_memory_size() {
        DCHECK(_bf_parsed);
        return _bf->size();
    }

    uint64_t get_memory_size() {
        DCHECK(_index_parsed);
        return _index_reader->get_memory_size();
    }

    friend class HashPrimaryKeyIndexIterator;

private:
    bool _index_parsed;
    bool _bf_parsed;
    std::unique_ptr<segment_v2::IndexedColumnReader> _index_reader;
    std::unique_ptr<segment_v2::BloomFilter> _bf;
    std::unique_ptr<segment_v2::CuckooIndexReader> _cuckoo_index_reader;
    std::unique_ptr<segment_v2::CuckooTable> _cuckoo_table;
};

class HashPrimaryKeyIndexIterator {
public:
    HashPrimaryKeyIndexIterator(const PrimaryKeyIndexReader* reader) : _reader(reader) {
        _reader->new_iterator(&_iter);
        auto index_type = vectorized::DataTypeFactory::instance().create_data_type(
                _reader->type_info()->type(), 1, 0);
        _column = index_type->create_column();
    }

    Status seek_at(Slice key) {
        DCHECK(_reader->_index_parsed);
        for (auto row : _reader->_cuckoo_table->find(key)) {
            bool match = false;
            RETURN_IF_ERROR(_do_check(key, row, match));
            if (match) {
                current_ordinal = row;
                seeked = true;
                return Status::OK();
            }
        }
        seeked = true;
        return Status::NotFound("key not found");
    }

    [[nodiscard]] segment_v2::rowid_t get_current_ordinal() const {
        DCHECK(seeked);
        return current_ordinal;
    }

    Status get_full_key(std::string& key) {
        DCHECK(seeked);
        Slice key_found;
        RETURN_IF_ERROR(_read_full_key(current_ordinal, &key_found));
        return Status::OK();
    }

private:
    Status _do_check(Slice const& key, segment_v2::rowid_t rid, bool& match) {
        Slice key_found;
        RETURN_IF_ERROR(_read_full_key(rid, &key_found));
        match = key == key_found;
        return Status::OK();
    }

    Status _read_full_key(segment_v2::rowid_t rid, Slice* key) {
        RETURN_IF_ERROR(_iter->seek_to_ordinal(rid));
        size_t num_to_read = 1;
        RETURN_IF_ERROR(_iter->next_batch(&num_to_read, _column));
        DCHECK(num_to_read == 1);
        *key = _column->get_data_at(0).to_slice();
        return Status::OK();
    }

    std::unique_ptr<segment_v2::IndexedColumnIterator> _iter;
    vectorized::MutableColumnPtr _column;
    bool seeked {false};
    segment_v2::rowid_t current_ordinal;
    const PrimaryKeyIndexReader* _reader;
};

} // namespace doris
