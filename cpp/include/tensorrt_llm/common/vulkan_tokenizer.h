// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TLLM_VULKAN_TOKENIZER_H
#define TLLM_VULKAN_TOKENIZER_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace tllm::vk
{

// Minimal GGUF header parser — reads vocab tokens and merges for BPE tokenization.
class GgufTokenizer
{
public:
    GgufTokenizer();
    ~GgufTokenizer();

    // Non-copyable
    GgufTokenizer(GgufTokenizer const&) = delete;
    GgufTokenizer& operator=(GgufTokenizer const&) = delete;

    // Load vocab from a GGUF file. Returns true on success.
    bool load(const std::string& gguf_path);

    // Tokenize text into BPE token IDs.
    std::vector<int32_t> encode(const std::string& text) const;

    // Detokenize token IDs back to text.
    std::string decode(const std::vector<int32_t>& token_ids) const;

    // Accessors
    int32_t vocab_size() const { return static_cast<int32_t>(m_vocab_raw.size()); }
    int32_t bos_token_id() const { return m_bos_token_id; }
    int32_t eos_token_id() const { return m_eos_token_id; }

private:
    struct TokenScore
    {
        std::string token;
        float score;
        int32_t id;
    };

    // BPE merge pair
    struct MergePair
    {
        int32_t first;   // token ID
        int32_t second;  // token ID
        int32_t merged;  // resulting token ID
    };

    std::vector<TokenScore> m_vocab_raw;
    std::unordered_map<std::string, int32_t> m_vocab_map;
    std::vector<MergePair> m_merges;

    int32_t m_bos_token_id = 151643;
    int32_t m_eos_token_id = 151643;

    // GGUF reader helpers
    struct GgufField
    {
        std::string key;
        int32_t type;
        std::vector<uint8_t> data;
        std::vector<std::string> string_parts;
    };

    std::vector<GgufField> m_fields;

    bool parse_gguf(const std::string& path);
    std::string get_field_string(const std::string& key) const;
    std::vector<std::string> get_field_string_array(const std::string& key) const;
    std::vector<int32_t> get_field_int_array(const std::string& key) const;

    // BPE internals
    struct WordBpeState
    {
        std::vector<int32_t> tokens;
    };

    // Encode a single word (no leading space) into BPE tokens
    std::vector<int32_t> bpe_encode_word(const std::string& word) const;

    // Simple regex pre-tokenizer: splits text into word chunks (GPT2-style)
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // Byte-to-unicode mapping (GPT-2 style)
    static std::unordered_map<uint8_t, char> build_byte_map();
};

} // namespace tllm::vk

#endif // TLLM_VULKAN_TOKENIZER_H
