// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_llm/common/vulkan_tokenizer.h"
#include "tensorrt_llm/common/vulkanCommon.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

namespace tllm::vk
{

GgufTokenizer::GgufTokenizer() = default;
GgufTokenizer::~GgufTokenizer() = default;

// ---- GGUF file format constants ----
// See: https://github.com/ggerganov/ggml/blob/master/gguf-py/gguf/gguf.py
// GGUF type IDs (from gguf.h / GGUF spec)
static const int32_t GGUF_TYPE_UINT8   = 0;
static const int32_t GGUF_TYPE_INT8    = 1;
static const int32_t GGUF_TYPE_UINT16  = 2;
static const int32_t GGUF_TYPE_INT16   = 3;
static const int32_t GGUF_TYPE_UINT32  = 4;
static const int32_t GGUF_TYPE_INT32   = 5;
static const int32_t GGUF_TYPE_FLOAT32 = 6;
static const int32_t GGUF_TYPE_BOOL    = 7;
static const int32_t GGUF_TYPE_STRING  = 8;
static const int32_t GGUF_TYPE_ARRAY   = 9;
static const int32_t GGUF_TYPE_UINT64  = 10;
static const int32_t GGUF_TYPE_INT64   = 11;
static const int32_t GGUF_TYPE_FLOAT64 = 12;

static const char* GGUF_MAGIC = "GGUF";

bool GgufTokenizer::parse_gguf(const std::string& path)
{
    FILE* f = nullptr;
#ifdef _MSC_VER
    fopen_s(&f, path.c_str(), "rb");
#else
    f = std::fopen(path.c_str(), "rb");
#endif
    if (!f)
    {
        TLLM_LOG_ERROR("Cannot open GGUF file: %s", path.c_str());
        return false;
    }

    m_fields.clear();

    // Read magic (4 bytes) + version (4 bytes) + header tensor count (8 bytes) + metadata count (8 bytes)
    char magic[4];
    if (std::fread(magic, 1, 4, f) != 4)
    {
        std::fclose(f);
        return false;
    }

    if (std::memcmp(magic, GGUF_MAGIC, 4) != 0)
    {
        TLLM_LOG_ERROR("Not a valid GGUF file (bad magic)");
        std::fclose(f);
        return false;
    }

    uint32_t version;
    if (std::fread(&version, sizeof(uint32_t), 1, f) != 1)
    {
        std::fclose(f);
        return false;
    }

    uint64_t tensor_count;
    if (std::fread(&tensor_count, sizeof(uint64_t), 1, f) != 1)
    {
        std::fclose(f);
        return false;
    }

    uint64_t metadata_count;
    if (std::fread(&metadata_count, sizeof(uint64_t), 1, f) != 1)
    {
        std::fclose(f);
        return false;
    }

    // Parse metadata fields
    for (uint64_t i = 0; i < metadata_count; i++)
    {
        GgufField field;

        // Read key length (uint64_t in GGUF spec)
        uint64_t key_len;
        if (std::fread(&key_len, sizeof(uint64_t), 1, f) != 1)
        {
            std::fclose(f);
            return false;
        }
        field.key.resize(key_len);
        if (key_len > 0 && std::fread(&field.key[0], 1, key_len, f) != key_len)
        {
            std::fclose(f);
            return false;
        }

        // Read value type
        uint32_t val_type;
        if (std::fread(&val_type, sizeof(uint32_t), 1, f) != 1)
        {
            std::fclose(f);
            return false;
        }
        field.type = static_cast<int32_t>(val_type);

        // Read value based on type
        if (val_type == GGUF_TYPE_STRING) // GGUF_TYPE_STRING
        {
            uint64_t str_len;
            if (std::fread(&str_len, sizeof(uint64_t), 1, f) != 1)
            {
                std::fclose(f);
                return false;
            }
            field.data.resize(str_len);
            if (str_len > 0 && std::fread(field.data.data(), 1, str_len, f) != str_len)
            {
                std::fclose(f);
                return false;
            }
            std::string s(reinterpret_cast<const char*>(field.data.data()), str_len);
            field.string_parts.push_back(s);
        }
        else if (val_type == GGUF_TYPE_ARRAY) // GGUF_TYPE_ARRAY
        {
            uint32_t array_type;
            if (std::fread(&array_type, sizeof(uint32_t), 1, f) != 1)
            {
                std::fclose(f);
                return false;
            }

            uint64_t array_len;
            if (std::fread(&array_len, sizeof(uint64_t), 1, f) != 1)
            {
                std::fclose(f);
                return false;
            }
            field.data.resize(array_len);

            for (uint64_t j = 0; j < array_len; j++)
            {
                if (array_type == GGUF_TYPE_STRING) // GGUF_TYPE_STRING
                {
                    uint64_t str_len;
                    if (std::fread(&str_len, sizeof(uint64_t), 1, f) != 1)
                    {
                        std::fclose(f);
                        return false;
                    }
                    std::string s(str_len, '\0');
                    if (str_len > 0 && std::fread(&s[0], 1, str_len, f) != str_len)
                    {
                        std::fclose(f);
                        return false;
                    }
                    field.string_parts.push_back(s);
                }
                else if (array_type == GGUF_TYPE_INT32)
                {
                    int32_t val;
                    if (std::fread(&val, sizeof(int32_t), 1, f) != 1)
                    {
                        std::fclose(f);
                        return false;
                    }
                    field.data.push_back(static_cast<uint8_t>(val & 0xFF));
                    field.data.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
                    field.data.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
                    field.data.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
                }
                else
                {
                    // Read and discard for scalar types
                    uint8_t val;
                    if (std::fread(&val, 1, 1, f) != 1)
                    {
                        std::fclose(f);
                        return false;
                    }
                }
            }
        }
        else if (val_type == GGUF_TYPE_BOOL)
        {
            uint8_t b;
            if (std::fread(&b, 1, 1, f) != 1)
            {
                std::fclose(f);
                return false;
            }
            field.data.push_back(b);
        }
        else if (val_type == GGUF_TYPE_UINT32 || val_type == GGUF_TYPE_INT32)
        {
            uint32_t val;
            if (std::fread(&val, sizeof(uint32_t), 1, f) != 1)
            {
                std::fclose(f);
                return false;
            }
            field.data.resize(4);
            field.data[0] = static_cast<uint8_t>(val & 0xFF);
            field.data[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
            field.data[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
            field.data[3] = static_cast<uint8_t>((val >> 24) & 0xFF);
        }
        else if (val_type == GGUF_TYPE_FLOAT32)
        {
            float val = 0.0f;
            if (std::fread(&val, sizeof(float), 1, f) != 1)
            {
                std::fclose(f);
                return false;
            }
            field.data.resize(4);
            std::memcpy(field.data.data(), &val, 4);
        }
        else if (val_type == GGUF_TYPE_UINT8)
        {
            uint8_t val;
            if (std::fread(&val, 1, 1, f) != 1)
            {
                std::fclose(f);
                return false;
            }
            field.data.push_back(val);
        }
        else if (val_type == GGUF_TYPE_INT8)
        {
            int8_t val;
            if (std::fread(&val, 1, 1, f) != 1)
            {
                std::fclose(f);
                return false;
            }
            field.data.push_back(static_cast<uint8_t>(val));
        }
        else
        {
            TLLM_LOG_WARNING("Unknown GGUF value type %u, skipping field", val_type);
            // Can't skip — we don't know the size. Abort.
            std::fclose(f);
            return false;
        }

        m_fields.push_back(std::move(field));
    }

    std::fclose(f);
    return true;
}

bool GgufTokenizer::load(const std::string& gguf_path)
{
    if (!parse_gguf(gguf_path))
    {
        return false;
    }

    // Extract vocab tokens: tokenizer.ggml.tokens (array of strings)
    auto tokens = get_field_string_array("tokenizer.ggml.tokens");
    if (tokens.empty())
    {
        TLLM_LOG_ERROR("No tokenizer.ggml.tokens found in GGUF");
        return false;
    }

    m_vocab_raw.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++)
    {
        m_vocab_raw[i].token = tokens[i];
        m_vocab_raw[i].score = static_cast<float>(tokens.size() - i);
        m_vocab_raw[i].id = static_cast<int32_t>(i);
        m_vocab_map[tokens[i]] = static_cast<int32_t>(i);
    }

    // Extract merges: tokenizer.ggml.merges (array of strings, format "tok1 tok2")
    auto merges = get_field_string_array("tokenizer.ggml.merges");
    for (const auto& m : merges)
    {
        // Merge pair format: "token1 token2" separated by space
        size_t space = m.find(' ');
        if (space == std::string::npos)
            continue;

        std::string t1 = m.substr(0, space);
        std::string t2 = m.substr(space + 1);

        auto it1 = m_vocab_map.find(t1);
        auto it2 = m_vocab_map.find(t2);
        if (it1 == m_vocab_map.end() || it2 == m_vocab_map.end())
            continue;

        // Merged token = t1 + t2
        std::string merged = t1 + t2;
        auto itm = m_vocab_map.find(merged);
        if (itm == m_vocab_map.end())
            continue;

        m_merges.push_back({it1->second, it2->second, itm->second});
    }

    // Read special token IDs
    auto bos_arr = get_field_int_array("tokenizer.ggml.bos_token_id");
    if (!bos_arr.empty())
        m_bos_token_id = static_cast<int32_t>(bos_arr[0]);

    auto eos_arr = get_field_int_array("tokenizer.ggml.eos_token_id");
    if (!eos_arr.empty())
        m_eos_token_id = static_cast<int32_t>(eos_arr[0]);

    TLLM_LOG_INFO("Tokenizer loaded: vocab_size=%zu, merges=%zu, bos=%d, eos=%d",
                  m_vocab_raw.size(), m_merges.size(), m_bos_token_id, m_eos_token_id);

    return true;
}

std::string GgufTokenizer::get_field_string(const std::string& key) const
{
    for (const auto& f : m_fields)
    {
        if (f.key == key)
        {
            if (!f.string_parts.empty())
                return f.string_parts[0];
            if (!f.data.empty())
                return std::string(reinterpret_cast<const char*>(f.data.data()), f.data.size());
        }
    }
    return "";
}

std::vector<std::string> GgufTokenizer::get_field_string_array(const std::string& key) const
{
    for (const auto& f : m_fields)
    {
        if (f.key == key)
            return f.string_parts;
    }
    return {};
}

std::vector<int32_t> GgufTokenizer::get_field_int_array(const std::string& key) const
{
    std::vector<int32_t> result;
    for (const auto& f : m_fields)
    {
        if (f.key == key)
        {
            for (size_t i = 0; i + 3 < f.data.size(); i += 4)
            {
                int32_t val = static_cast<int32_t>(
                    static_cast<uint32_t>(f.data[i]) |
                    (static_cast<uint32_t>(f.data[i + 1]) << 8) |
                    (static_cast<uint32_t>(f.data[i + 2]) << 16) |
                    (static_cast<uint32_t>(f.data[i + 3]) << 24));
                result.push_back(val);
            }
        }
    }
    return result;
}

// ---- Byte-to-unicode mapping (GPT-2 style) ----
std::unordered_map<uint8_t, char> GgufTokenizer::build_byte_map()
{
    static std::unordered_map<uint8_t, char> byte_map;
    if (!byte_map.empty())
        return byte_map;

    // GPT-2 byte-to-unicode mapping
    auto construct = []() -> std::unordered_map<uint8_t, char> {
        std::unordered_map<uint8_t, char> m;
        // Letters, digits, space
        for (int c = 'a'; c <= 'z'; c++) m[static_cast<uint8_t>(c)] = static_cast<char>(c - 'a');
        for (int c = 'A'; c <= 'Z'; c++) m[static_cast<uint8_t>(c)] = static_cast<char>(c - 'A' + 26);
        for (int c = '0'; c <= '9'; c++) m[static_cast<uint8_t>(c)] = static_cast<char>(c - '0' + 52);

        // Special characters
        std::string special = "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~ ";
        int offset = 62;
        for (char c : special)
            m[static_cast<uint8_t>(c)] = static_cast<char>(offset++);

        // Bytes 0-32 and 127+ get mapped to unicode range starting from 0
        offset = 62 + static_cast<int>(special.size());
        for (int b = 0; b < 256; b++)
        {
            uint8_t ub = static_cast<uint8_t>(b);
            if (m.find(ub) == m.end())
            {
                m[ub] = static_cast<char>(offset++);
            }
        }
        return m;
    }();

    byte_map = construct;
    return byte_map;
}

// ---- Pre-tokenization: GPT2 BPE pattern ----
std::vector<std::string> GgufTokenizer::pre_tokenize(const std::string& text) const
{
    std::vector<std::string> tokens;

    // GPT2 pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    // We approximate with a simpler regex since C++ std::regex doesn't do \p{}
    size_t i = 0;
    size_t n = text.size();

    while (i < n)
    {
        // Check for contractions
        if (i + 1 < n && text[i] == '\'')
        {
            if (text[i + 1] == 's') { tokens.push_back("'s"); i += 2; continue; }
            if (text[i + 1] == 't') { tokens.push_back("'t"); i += 2; continue; }
            if (i + 2 < n && text[i + 1] == 'r' && text[i + 2] == 'e') { tokens.push_back("'re"); i += 3; continue; }
            if (i + 2 < n && text[i + 1] == 'v' && text[i + 2] == 'e') { tokens.push_back("'ve"); i += 3; continue; }
            if (text[i + 1] == 'm') { tokens.push_back("'m"); i += 2; continue; }
            if (text[i + 1] == 'l' && i + 2 < n && text[i + 2] == 'l') { tokens.push_back("'ll"); i += 3; continue; }
            if (text[i + 1] == 'd') { tokens.push_back("'d"); i += 2; continue; }
        }

        // Match word (letters + digits)
        if (std::isalpha(static_cast<unsigned char>(text[i])) || std::isdigit(static_cast<unsigned char>(text[i])))
        {
            // Include leading spaces
            size_t start = i;
            if (i > 0 && text[i - 1] == ' ' && !tokens.empty())
            {
                // Prepend space to word (byte-level: space is already in vocab)
            }
            size_t end = i;
            while (end < n && (std::isalnum(static_cast<unsigned char>(text[end]))))
            {
                end++;
            }
            // Include leading space in the token
            if (i > 0 && text[i - 1] == ' ')
                tokens.push_back(" " + text.substr(i, end - i));
            else
                tokens.push_back(text.substr(i, end - i));
            i = end;
            continue;
        }

        // Match whitespace-only runs (trailing spaces)
        if (std::isspace(static_cast<unsigned char>(text[i])))
        {
            size_t start = i;
            while (i < n && std::isspace(static_cast<unsigned char>(text[i])))
                i++;
            // trailing spaces before a word — include with next token
            // but if next is non-word, emit
            if (i >= n || !std::isalnum(static_cast<unsigned char>(text[i])))
            {
                tokens.push_back(text.substr(start, i - start));
            }
            continue;
        }

        // Punctuation / non-alphanumeric
        size_t start = i;
        while (i < n && !std::isalnum(static_cast<unsigned char>(text[i])) &&
               !std::isspace(static_cast<unsigned char>(text[i])))
        {
            i++;
        }
        tokens.push_back(text.substr(start, i - start));
    }

    return tokens;
}

// ---- BPE encode a single word ----
std::vector<int32_t> GgufTokenizer::bpe_encode_word(const std::string& word) const
{
    // Convert word using byte-to-unicode mapping
    auto byte_map = build_byte_map();

    // For GPT2 BPE, the vocab is keyed on bytes
    // First, try to find the whole word in vocab
    std::string mapped;
    mapped.reserve(word.size());
    for (unsigned char c : word)
    {
        auto it = byte_map.find(c);
        if (it != byte_map.end())
            mapped += it->second;
    }

    // Try direct vocab match
    auto it = m_vocab_map.find(word);
    if (it != m_vocab_map.end())
        return {it->second};

    // Try mapped version
    it = m_vocab_map.find(mapped);
    if (it != m_vocab_map.end())
        return {it->second};

    // Fall back to per-character byte representation
    // GPT2 uses \x01 for byte 1, etc. in vocab
    std::vector<int32_t> result;
    for (unsigned char c : word)
    {
        // Try the raw byte as a 1-char token
        std::string one_char(1, static_cast<char>(c));
        auto it2 = m_vocab_map.find(one_char);
        if (it2 != m_vocab_map.end())
            result.push_back(it2->second);
        else
        {
            // Try GPT2 byte fallback: \xNN format
            char hex_buf[5];
            std::snprintf(hex_buf, sizeof(hex_buf), "\\x%02x", c);
            std::string hex_token(hex_buf);
            auto it3 = m_vocab_map.find(hex_token);
            if (it3 != m_vocab_map.end())
                result.push_back(it3->second);
            else
            {
                // Unknown — use unk (0)
                result.push_back(0);
            }
        }
    }

    if (result.empty())
        result.push_back(0);

    return result;
}

std::vector<int32_t> GgufTokenizer::encode(const std::string& text) const
{
    std::vector<int32_t> tokens;

    // Add BOS
    tokens.push_back(m_bos_token_id);

    auto words = pre_tokenize(text);
    for (const auto& word : words)
    {
        auto word_tokens = bpe_encode_word(word);
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
    }

    return tokens;
}

std::string GgufTokenizer::decode(const std::vector<int32_t>& token_ids) const
{
    std::string result;
    for (int32_t tid : token_ids)
    {
        if (tid < 0 || static_cast<size_t>(tid) >= m_vocab_raw.size())
        {
            result += "<" + std::to_string(tid) + ">";
            continue;
        }
        result += m_vocab_raw[tid].token;
    }

    // Clean up: GPT2 BPE uses byte-level mapping, but for simplicity we return raw
    return result;
}

} // namespace tllm::vk
