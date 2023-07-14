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
#include <glog/logging.h>
#include <string.h>

#include <climits>
#include <coroutine>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>

#include "common/compiler_util.h"
#include "common/global_types.h"
#include "common/status.h"
#include "olap/olap_common.h"
#include "olap/rowset/segment_v2/common.h"
#include "olap/utils.h"
#include "util/murmur_hash3.h"
#include "util/slice.h"
#include "vec/common/string_ref.h"

namespace doris {
namespace segment_v2 {

struct CuckooTableOptions {
    double max_density = 0.80;
    uint32_t layer_size = 64;
    uint32_t max_search_nodes = 1024;
    uint32_t num_slot_per_batch = 1024;
    bool read_only = false;
    bool do_shrink_to_fit = true;
};

using CuckooValue = rowid_t;
using CuckooFinger = uint32_t;
using CuckooHashFunc = std::function<void(const void*, const int, uint32_t, void*)>;

enum CuckooLiterals : int {
    EMPTY_SLOT_VALUE = 0,
    EMPTY_SLOT_FINGER = 0,
    DEFAULT_NOT_EMPTY_SLOT_FINGER = 1
};

struct CuckooTableStat {
    void show() {
        LOG(INFO) << fmt::format("Cuckoo Statics: insert: {} us, find: {} us, load: {} us",
                                 insert_watch, find_watch, load_watch);
        reset();
    }
    void reset() {
        insert_watch = 0;
        find_watch = 0;
        load_watch = 0;
    }
    static CuckooTableStat* get_stat() { return &stat; }
    uint64_t insert_watch {0};
    uint64_t find_watch {0};
    uint64_t load_watch {0};
    static CuckooTableStat stat;
};

class CuckooHashCodeGenerator {
public:
    // Default seed for cuckoo hash function. It comes from date +%s.
    constexpr static uint32_t CUCKOO_HASH_SEED = 1685524199;

    [[nodiscard]] static uint32_t hash(uint32_t key, uint32_t scalar = 0) {
        uint32_t result;
        murmur_hash3_x86_32(reinterpret_cast<const char*>(&key), sizeof(key),
                            scalar * CUCKOO_HASH_SEED, static_cast<void*>(&result));
        return result;
    }

    [[nodiscard]] static uint32_t hash(Slice key, uint32_t scalar = 0) {
        uint32_t result;
        murmur_hash3_x86_32(key.get_data(), key.get_size(), scalar * CUCKOO_HASH_SEED,
                            static_cast<void*>(&result));
        return result;
    }

    [[nodiscard]] static CuckooFinger get_finger(Slice key) {
        CuckooFinger finger = hash(key);
        return UNLIKELY(finger == EMPTY_SLOT_FINGER) ? DEFAULT_NOT_EMPTY_SLOT_FINGER : finger;
    }
};

// Base class for cuckoo table
template <int num_layers>
class FixedCuckooTable {
public:
    struct Slot {
    public:
        Slot() : Slot(EMPTY_SLOT_FINGER, EMPTY_SLOT_VALUE) {}
        Slot(CuckooFinger finger, CuckooValue value) : _finger(finger), _value(value) {}
        [[nodiscard]] bool is_empty() const { return _finger == EMPTY_SLOT_FINGER; }
        [[nodiscard]] bool is_occupied() const { return _finger != EMPTY_SLOT_FINGER; }
        [[nodiscard]] CuckooValue value() const { return _value; }
        [[nodiscard]] CuckooFinger finger() const { return _finger; }

        friend class FixedCuckooTable;

    private:
        CuckooFinger _finger;
        CuckooValue _value;
    };

    FixedCuckooTable() = default;
    ~FixedCuckooTable() = default;
    FixedCuckooTable(const FixedCuckooTable&) = default;
    FixedCuckooTable(FixedCuckooTable&&) = default;
    FixedCuckooTable& operator=(FixedCuckooTable const&) = default;
    FixedCuckooTable& operator=(FixedCuckooTable&&) = default;

    Status init(CuckooTableOptions opts) {
        _len_table = opts.layer_size;
        _max_density = opts.max_density;
        _max_search_nodes = opts.max_search_nodes;
        _num_slot_per_batch = opts.num_slot_per_batch;
        _do_shrink_to_fit = opts.do_shrink_to_fit;
        _read_only = opts.read_only;

        RETURN_IF_ERROR(_check_arguments());

        if (_len_table) {
            _table = std::vector<Slot>(num_layers * _len_table);
            if (!_read_only) {
                _visited = std::vector<bool>(num_layers * _len_table);
            }
        }
        return Status::OK();
    }

    Status init(FixedCuckooTable const& other, uint32_t layer_size) {
        _len_table = layer_size;
        num_layers = other.num_layers;
        _max_density = other._max_density;
        _max_search_nodes = other._max_search_nodes;
        _num_slot_per_batch = other._num_slot_per_batch;
        _do_shrink_to_fit = other._do_shrink_to_fit;
        _read_only = other._read_only;

        RETURN_IF_ERROR(_check_arguments());

        if (_len_table) {
            _table = std::vector<Slot>(num_layers * _len_table);
            if (!_read_only) {
                _visited = std::vector<bool>(num_layers * _len_table);
            }
        }
        return Status::OK();
    }

    Status insert(Slice key, CuckooValue value) {
        if (UNLIKELY(_occupied >= static_cast<uint32_t>(ceil(capacity() * _max_density)))) {
            RETURN_IF_ERROR(_expand());
        }

        auto cur_finger = CuckooHashCodeGenerator::get_finger(key);
        auto cur_value = value;
        while (true) {
            auto st = _insert(cur_finger, cur_value);
            if (st.ok()) {
                break;
            }
            RETURN_IF_ERROR(_expand());
        }
        return Status::OK();
    }

    struct FindCoroutine {
        struct promise_type {
            auto get_return_object() { return FindCoroutine {Handle::from_promise(*this)}; }
            auto initial_suspend() noexcept { return std::suspend_always {}; }
            auto final_suspend() noexcept { return std::suspend_always {}; }
            void unhandled_exception() { std::terminate(); }
            auto yield_value(std::pair<Status, CuckooValue> const& value) {
                current_value = value;
                return std::suspend_always {};
            }
            std::pair<Status, CuckooValue> current_value;
        };
        using Handle = std::coroutine_handle<promise_type>;
        explicit FindCoroutine(Handle handle) : coro(handle) {}
        ~FindCoroutine() {
            if (coro) {
                coro.destroy();
            }
        }
        bool next() {
            if (!coro.done()) {
                coro.resume();
            }
            return !coro.done() && coro.promise().current_value.first.ok();
        }
        [[nodiscard]] uint32_t get() const { return coro.promise().current_value.second; }

    private:
        Handle coro;
    };

    [[nodiscard]] FindCoroutine find_coro(Slice key) const {
        auto finger = CuckooHashCodeGenerator::get_finger(key);
        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            auto hash_code = CuckooHashCodeGenerator::hash(finger, layer);
            auto slot = hash_code % _len_table;
            if (UNLIKELY(finger == _table[_offset(layer, slot)].finger())) {
                co_yield {Status::OK(), _table[_offset(layer, slot)].value()};
            }
        }
        co_yield {Status::NotFound("key not found"), {}};
    }

    [[nodiscard]] std::vector<CuckooValue> find(Slice key) const {
        OlapStopWatch watch;
        auto finger = CuckooHashCodeGenerator::get_finger(key);
        std::vector<CuckooValue> ret;
        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            auto hash_code = CuckooHashCodeGenerator::hash(finger, layer);
            auto slot = hash_code % _len_table;
            if (UNLIKELY(finger == _table[_offset(layer, slot)].finger())) {
                ret.emplace_back(_table[_offset(layer, slot)].value());
            }
        }
        CuckooTableStat::get_stat()->find_watch += watch.get_elapse_time_us();
        return ret;
    }

    Status shrink_to_fit() {
        uint32_t optimal_length = std::max(64u, _min_length());
        RETURN_IF_ERROR(resize(optimal_length));
        return Status::OK();
    }

    [[nodiscard]] uint32_t capacity() const { return _len_table * num_layers; }

    Status resize(uint32_t len_table) {
        if (!_is_length_ok(len_table)) {
            auto msg = fmt::format(
                    "Cuckoo table length is eithor too large or too small, "
                    "length should be within the interval [{}, {}], but now "
                    "num_layers: {}, length: {}, occupied: {}, ",
                    _min_length(), _max_length(), num_layers, len_table, _occupied);
            LOG(INFO) << msg;
            return Status::InvalidArgument(msg);
        }
        FixedCuckooTable<num_layers> new_table;
        RETURN_IF_ERROR(new_table.init(*this, len_table));
        for (auto [finger, value] : _table) {
            if (UNLIKELY(finger == EMPTY_SLOT_FINGER)) {
                continue;
            }
            auto st = new_table._insert(finger, value);
            if (st.template is<ErrorCode::NOT_FOUND>()) {
                LOG(INFO) << fmt::format(
                        "Cuckoo table is doing a recurrently insert when try to resize to {}",
                        len_table);
                return Status::Corruption(
                        "Cuckoo table resize failed, "
                        "cuz cuckoo table is doing a recurrently insert");
            }
        }
        LOG(INFO) << fmt::format("Cuckoo table resize to {} * {} successfully.", len_table,
                                 num_layers);
        *this = std::move(new_table);
        return Status::OK();
    }

    [[nodiscard]] uint32_t layers() const { return num_layers; }

    [[nodiscard]] size_t size() const { return num_layers * _len_table * sizeof(Slot); }

    friend class CuckooIndexWriter;
    friend class CuckooIndexReader;

private:
    [[nodiscard]] size_t _block_size() const { return _num_slot_per_batch * sizeof(Slot); }

    uint32_t get_block_id(uint32_t layer, uint32_t slot) {
        auto slot_offset = _offset(layer, slot);
        auto batch_id = slot_offset / _num_slot_per_batch;
        return batch_id;
    }

    size_t get_block_offset(uint32_t layer, uint32_t slot) {
        return get_block_id(layer, slot) * _block_size();
    }

    Slice get_block(uint32_t layer, uint32_t slot) {
        auto batch_offset = get_block_offset(layer, slot);
        return Slice(reinterpret_cast<const uint8_t*>(batch_offset),
                     std::min(size() - batch_offset, _block_size()));
    }

    Status set_block(uint32_t layer, uint32_t slot, Slice src_data) {
        auto dest_data = get_block(layer, slot);
        if (UNLIKELY(src_data.get_size() != src_data.get_size())) {
            return Status::InvalidArgument("src_data's data size should be equal to this block");
        }
        std::copy(src_data.get_data(), src_data.get_data() + src_data.get_size(),
                  const_cast<char*>(dest_data.get_data()));
        return Status::OK();
    }

    Status set_layer(uint32_t layer, Slice value) {
        if (UNLIKELY(_len_table * sizeof(Slot) != value.get_size())) {
            return Status::InvalidArgument("value's size should be equal to the layer");
        }
        std::copy(value.get_data(), value.get_data() + value.get_size(),
                  reinterpret_cast<char*>(_table.data()) + layer * _len_table * sizeof(Slot));
        return Status::OK();
    }

    [[nodiscard]] uint32_t _offset(uint32_t layer, uint32_t slot) const {
        DCHECK(layer < num_layers && slot < _len_table) << fmt::format(
                "Invalid Arguments: layer({}) vs num_layers({}), slot({}) vs len_table({})", layer,
                num_layers, slot, _len_table);
        return layer * _len_table + slot;
    }

    [[nodiscard]] bool _is_length_ok(uint32_t len_table) const {
        // if `len_table * num_layers * sizeof(CuckooSlotItem) > MAXIMUM_BYTES`,
        // the table will be too large, so we assume it's not ok.
        // if `_occupied > len_table * num_layers * _max_density`,
        // it's not proper to insert so many keys into the table,
        // so we assume it's not ok.
        return len_table >= _min_length() && len_table <= _max_length();
    }

    Status _expand() {
        uint32_t new_length = _len_table << 1;
        while (true) {
            auto st = resize(new_length);
            if (st.ok()) {
                break;
            }
            if (st.template is<ErrorCode::CORRUPTION>()) {
                new_length = _len_table << 1;
                continue;
            }
            return st;
        }
        return Status::OK();
    }

    struct SearchTreeNode {
        int prev_node_id {-1};
        int layer_id {-1};
        int slot_id {-1};
    };

    void _do_kick(Slot const& slot, uint32_t cur_node_id,
                  std::vector<SearchTreeNode> const& search_tree) {
        if (UNLIKELY(cur_node_id + 1 >= _max_search_nodes)) {
            LOG(INFO) << fmt::format(
                    "Cuckoo table is going on a vary long kick path when inserting "
                    "finger: {}, value: {}, length: {}",
                    slot.finger(), slot.value(), cur_node_id + 1);
        }
        auto [prev_node_id, cur_layer, cur_slot] = search_tree[cur_node_id];
        while (prev_node_id != -1) {
            auto [prev_prev_node_id, prev_layer_id, prev_slot_id] = search_tree[prev_node_id];
            _table[_offset(cur_layer, cur_slot)] = _table[_offset(prev_layer_id, prev_slot_id)];
            cur_node_id = prev_node_id;
            prev_node_id = prev_prev_node_id;
            cur_layer = prev_layer_id;
            cur_slot = prev_slot_id;
        }
        _table[_offset(cur_layer, cur_slot)] = slot;
        ++_occupied;
    }

    void _reset_visited_flag(std::vector<SearchTreeNode> const& search_tree) {
        for (auto [_, layer, slot] : search_tree) {
            _visited[_offset(layer, slot)] = false;
        }
    }

    Status _insert(CuckooFinger finger, CuckooValue value) {
        OlapStopWatch watch;
        std::vector<SearchTreeNode> search_tree;
        // initialize search tree
        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            auto hash_code = CuckooHashCodeGenerator::hash(finger, layer);
            auto slot_id = hash_code % _len_table;
            search_tree.emplace_back(-1, layer, slot_id);
            _visited[_offset(layer, slot_id)] = true;
            if (UNLIKELY(_table[_offset(layer, slot_id)].finger() == EMPTY_SLOT_FINGER)) {
                _do_kick({finger, value}, search_tree.size() - 1, search_tree);
                _reset_visited_flag(search_tree);
                CuckooTableStat::get_stat()->insert_watch += watch.get_elapse_time_us();
                return Status::OK();
            }
        };
        // use bfs to search for a empty slot to insert into
        for (uint32_t cur_node_id = 0; cur_node_id < search_tree.size(); ++cur_node_id) {
            auto [_, cur_layer, cur_slot] = search_tree[cur_node_id];
            auto cur_finger = _table[_offset(cur_layer, cur_slot)].finger();
            for (uint32_t nxt_layer = 0; nxt_layer < num_layers; ++nxt_layer) {
                auto hash_code = CuckooHashCodeGenerator::hash(cur_finger, nxt_layer);
                auto nxt_slot = hash_code % _len_table;
                if (_visited[_offset(nxt_layer, nxt_slot)]) {
                    continue;
                }
                search_tree.emplace_back(cur_node_id, nxt_layer, nxt_slot);
                _visited[_offset(nxt_layer, nxt_slot)] = true;
                if (UNLIKELY(_table[_offset(nxt_layer, nxt_slot)].finger() == EMPTY_SLOT_FINGER)) {
                    _do_kick({finger, value}, search_tree.size() - 1, search_tree);
                    _reset_visited_flag(search_tree);
                    CuckooTableStat::get_stat()->insert_watch += watch.get_elapse_time_us();
                    return Status::OK();
                }
            }
        }
        _reset_visited_flag(search_tree);
        CuckooTableStat::get_stat()->insert_watch += watch.get_elapse_time_us();
        return Status::NotFound("No empty slot to insert into");
    }

    Status _check_arguments() {
        if (!_is_length_ok(_len_table)) {
            auto msg = fmt::format(
                    "Cuckoo table length is eithor too large or too small, "
                    "length should be within the interval [{}, {}], but now "
                    "num_layers: {}, length: {}, occupied: {}, ",
                    _min_length(), _max_length(), num_layers, _len_table, _occupied);
            LOG(INFO) << msg;
            return Status::InvalidArgument(msg);
        }
        return Status::OK();
    }

    [[nodiscard]] uint32_t _min_length() const {
        return ceil(_occupied / _max_density / num_layers);
    }

    [[nodiscard]] uint32_t _max_length() const { return INT_MAX; }

    std::vector<Slot> _table;
    std::vector<bool> _visited;

    uint32_t _max_search_nodes {0};
    uint32_t _len_table {0};
    uint32_t _occupied {0};
    uint32_t _num_slot_per_batch {0};
    double _max_density {1.0};
    bool _do_shrink_to_fit {true};
    bool _read_only {false};
};

using CuckooTable = FixedCuckooTable<3>;

} // namespace segment_v2
} // namespace doris
