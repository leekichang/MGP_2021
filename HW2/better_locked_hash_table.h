#ifndef _BETTER_LOCKED_HASH_TABLE_H_
#define _BETTER_LOCKED_HASH_TABLE_H_

#define OA_TABLE_SIZE 10000

#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include "hash_table.h"
#include "linked_list.h"


class better_locked_hash_table : public hash_table
{
    // TODO
    typedef struct _table
    {
        int key;
        bool is_empty;
        _table()
        {
            key = INT32_MAX;
            is_empty = true;
        }
    } Table;
    std::mutex mutex_arr[OA_TABLE_SIZE];
    Table *table;
    // TODO

public:
    better_locked_hash_table()
    {
        // TODO
        this->table = new Table[OA_TABLE_SIZE];
        // TODO
    }

    bool contains(int key)
    {
        // TODO
        std::lock_guard<std::mutex> lock(mutex_arr[key % OA_TABLE_SIZE]);
        int hash_value = key % OA_TABLE_SIZE;
        int i = 1;
        while (table[hash_value + i].is_empty == false && table[hash_value + i].key < key && hash_value + i + 1 < OA_TABLE_SIZE)
        {
            ++i;
        }
        if (table[hash_value + i].is_empty != true && table[hash_value + i].key == key)
        {
            return true;
        }
        else
        {
            return false;
        }
        // TODO
    }

    bool insert(int key)
    {
        // TODO
        std::lock_guard<std::mutex> lock(mutex_arr[key % OA_TABLE_SIZE]);
        int hash_value = key % OA_TABLE_SIZE;
        int i = 1;
        while (table[hash_value + i].is_empty == false && table[hash_value + i].key < key && hash_value + i + 1 < OA_TABLE_SIZE)
        {
            ++i;
        }
        if (table[hash_value + i].is_empty == false && table[hash_value + i].key == key)
        {
            return false;
        }
        else if (hash_value + i + 1 >= OA_TABLE_SIZE)
        {
            return false;
        }
        else
        {
            table[hash_value + i].is_empty = false;
            table[hash_value + i].key = key;
            return true;
        }
        // TODO
    }

    bool remove(int key)
    {
        // TODO
        std::lock_guard<std::mutex> lock(mutex_arr[key % OA_TABLE_SIZE]);
        int hash_value = key % OA_TABLE_SIZE;
        int i = 1;

        if (table[hash_value].is_empty == true)
        {
            return false;
        }

        while (table[hash_value + i].is_empty == false && table[hash_value + i].key < key && hash_value + i + 1 < OA_TABLE_SIZE)
        {
            ++i;
        }

        if (table[hash_value + i].is_empty == false && table[hash_value + i].key == key)
        {
            table[hash_value + i].key = INT32_MAX;
            table->is_empty = true;
            return true;
        }
        else
        {
            return false;
        }

        // TODO
    }
};

#endif
