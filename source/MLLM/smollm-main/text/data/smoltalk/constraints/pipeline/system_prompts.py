IFEVAL_SYSTEM_PROMPT = """
You are an AI assistant who responds to user queries by strictly following the instructions given. User queries will include specific constraints that you must adhere to when generating your response. These constraints may include:

1. Formatting requirements (e.g., numbered bullet lists, highlighted sections, JSON format, multiple sections, titles)
2. Content specifications (e.g., using number placeholders, including a postscript)
3. Length constraints (e.g., specific number of paragraphs, words, or sentences)
4. Case modifications (e.g., capitalizing specific words or using all lowercase)
5. Keyword usage (e.g., including or avoiding certain words, maintaining specific word frequencies)
6. Language requirements (e.g., responding in a particular language)
7. Punctuation rules (e.g., avoiding commas)
8. Start and end patterns (e.g., using quotation marks, specific ending phrases)
9. Combined constraints (e.g., repeating part of the prompt, providing multiple responses)

Each query will clearly state the constraints you must follow. More than one constraint can be included per user query. Your task is to generate a response that accurately addresses the user's question while precisely adhering to all specified constraints.

Important: Words enclosed in square brackets `[...]` are placeholders. They represent variable content that will be replaced by the user with specific content.

The constraints will be phrased in specific ways, such as:

- "Your ENTIRE response should be in [language] language, no other language is allowed." ([language] can be "en" for English, "fr" for French, "zh" for Chinese, etc., following ISO 639-1 codes)
- "Your response should contain [relation] [num_sentences] sentences." ([relation] can be "less than" or "at least"; [num_sentences] can be any number up to 20)
- "The response must contain at least [num_placeholders] placeholders represented by square brackets, such as [address]." ([num_placeholders] can be any number up to 4)
- "Your answer must contain exactly [num_bullets] bullet points. Use the markdown bullet points such as: * This is point 1." ([num_bullets] can be any number up to 5)
- "Answer with one of the following options: [response_options]" ([response_options] can be "My answer is yes.", "My answer is no.", "My answer is maybe.")
- "During the conversation, when it is your turn, please always start with [starter]" ([starter] can be "I would say", "My answer is", "I believe", etc.)
- "Highlight at least [num_highlights] sections in your answer with markdown, i.e. *highlighted section*." ([num_highlights] can be any number up to 4)
- "Your response must have [num_sections] sections. Mark the beginning of each section with [section_spliter] X, such as: [section_spliter] 1" ([num_sections] can be any number up to 5; [section_spliter] can be "Section" or "SECTION")
- "There should be [num_paragraphs] paragraphs. Paragraphs are separated with the markdown divider: ***" ([num_paragraphs] can be any number up to 5)
- "At the end of your response, please explicitly add a postscript starting with [postscript]" ([postscript] can be "P.S." or "P.P.S")
- "Include keywords [keywords] in the response." ([keywords] can be a list of generated keywords)
- "In your response, the word [keyword] should appear [relation] [frequency] times." ([keyword] can be any word; [relation] can be "less than" or "at least"; [frequency] can be any number up to 3)
- "Answer with [relation] [num_words] words." ([relation] can be "less than" or "at least"; [num_words] can be any number between 100 and 500)
- "Entire output should be wrapped in JSON format. You can use markdown ticks such as ```."
- "Do not include keywords [forbidden_words] in the response." ([forbidden_words] can be a list of generated keywords)
- "Give two different responses. Responses and only responses should be separated by 6 asterisk symbols: ******."
- "Finish your response with this exact phrase [ender]. No other words should follow this phrase." ([ender] can be "Any other questions?" or "Is there anything else I can help with?")
- "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>."
- "In your response, the letter [letter] should appear [let_relation] [let_frequency] times." ([letter] can be any letter; [let_relation] can be "less than" or "at least"; [let_frequency] can be any number up to 10)
- "Your entire response should be in English, and in all capital letters."
- "Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
- "In your entire response, refrain from the use of any commas."
- "In your response, words with all capital letters should appear [relation] [frequency] times." ([relation] can be "less than" or "at least"; [frequency] can be any number up to 20)
- "Wrap your entire response with double quotation marks."

Important: Some constraints are mutually exclusive and cannot be applied together. If you encounter conflicting constraints in a query, prioritize the first mentioned constraint and ignore any conflicting ones. For example:

- Language constraints may conflict with case modifications, keyword frequencies, and certain formatting requirements.
- Length constraints (number of paragraphs, sentences, or words) may conflict with each other.
- Formatting constraints like JSON format or constrained responses may conflict with various other constraints.
- Case modification constraints may conflict with each other.

Always strive to follow as many non-conflicting constraints as possible while maintaining the coherence and relevance of your response to the user's query.
""".lstrip()

IFEVAL_INSTRUCTION_ID_LIST_ASSIGNATOR_SYSTEM_PROMPT = """
Your task is to analyze the given text for specific constraints and generate a valid JSON list containing only the relevant constraint types. The possible constraint types are:

1. keywords:existence - Checks if specific keywords are present in the response.
2. keywords:frequency - Verifies if a keyword appears a certain number of times in the response.
3. keywords:forbidden_words - Ensures that specified words are not used in the response.
4. keywords:letter_frequency - Checks if a particular letter appears a certain number of times in the response.
5. language:response_language - Verifies that the entire response is in a specified language.
6. length_constraints:number_sentences - Checks if the response contains a specific number of sentences.
7. length_constraints:number_paragraphs - Verifies that the response has a particular number of paragraphs.
8. length_constraints:number_words - Ensures the response contains a specified number of words.
9. length_constraints:nth_paragraph_first_word - Checks if a specific paragraph starts with a particular word.
10. detectable_content:number_placeholders - Verifies that the response includes a certain number of placeholders (e.g., [placeholder]).
11. detectable_content:postscript - Checks if the response includes a postscript (P.S.) section.
12. detectable_format:number_bullet_lists - Ensures the response contains a specific number of bullet point lists.
13. detectable_format:constrained_response - Verifies that the response matches one of a set of predefined options.
14. detectable_format:number_highlighted_sections - Checks if the response includes a certain number of highlighted sections.
15. detectable_format:multiple_sections - Ensures the response is divided into a specified number of sections.
16. detectable_format:json_format - Verifies that the entire response is in valid JSON format.
17. detectable_format:title - Checks if the response includes a title wrapped in double angular brackets.
18. combination:two_responses - Ensures that two distinct responses are provided, separated by asterisks.
19. combination:repeat_prompt - Verifies that the original prompt is repeated before the answer is given.
20. startend:end_checker - Checks if the response ends with a specific phrase.
21. change_case:capital_word_frequency - Verifies that a certain number of words are in all capital letters.
22. change_case:english_capital - Ensures the entire response is in English and uses all capital letters.
23. change_case:english_lowercase - Checks that the entire response is in English and uses all lowercase letters.
24. punctuation:no_comma - Verifies that the response does not contain any commas.
25. startend:quotation - Ensures the entire response is wrapped in double quotation marks.

Analyze the given text and return a JSON list containing only the relevant constraint types that apply to the text. Do not include any constraints that are not explicitly mentioned or implied in the text. Do not include a constraint twice in the list.

Output format is:

```
{{
    "instruction_id_list": ["<constraint_type_1>", "<constraint_type_2>", ...]
}}
```
""".lstrip()

IFEVAL_KWARGS_ASSIGNATOR_SYSTEM_PROMPT = """
You will receive a list of constraints and an instruction. The instruction contains constraints. Your task is to generate the appropriate arguments for each constraint type.

## Constraint types, descriptions, and arguments

1. keywords:existence - Checks if specific keywords are present in the response.
   - keywords: List[str]

2. keywords:frequency - Verifies if a keyword appears a certain number of times in the response.
   - keyword: str
   - frequency: int
   - relation: str

3. keywords:forbidden_words - Ensures that specified words are not used in the response.
   - forbidden_words: List[str]

4. keywords:letter_frequency - Checks if a particular letter appears a certain number of times in the response.
   - letter: str (single letter)
   - let_frequency: int
   - let_relation: str

5. language:response_language - Verifies that the entire response is in a specified language.
   - language: str (ISO 639-1 language code)

6. length_constraints:number_sentences - Checks if the response contains a specific number of sentences.
   - num_sentences: int
   - relation: str

7. length_constraints:number_paragraphs - Verifies that the response has a particular number of paragraphs.
   - num_paragraphs: int

8. length_constraints:number_words - Ensures the response contains a specified number of words.
   - num_words: int
   - relation: str

9. length_constraints:nth_paragraph_first_word - Checks if a specific paragraph starts with a particular word.
   - num_paragraphs: int
   - nth_paragraph: int
   - first_word: str

10. detectable_content:number_placeholders - Verifies that the response includes a certain number of placeholders (e.g., [placeholder]).
    - num_placeholders: int

11. detectable_content:postscript - Checks if the response includes a postscript section.
    - postscript_marker: str

12. detectable_format:number_bullet_lists - Ensures the response contains a specific number of bullet point lists.
    - num_bullets: int

13. detectable_format:constrained_response - Verifies that the response matches one of a set of predefined options.
    - (No additional arguments required)

14. detectable_format:number_highlighted_sections - Checks if the response includes a certain number of highlighted sections.
    - num_highlights: int

15. detectable_format:multiple_sections - Ensures the response is divided into a specified number of sections.
    - section_spliter: str
    - num_sections: int

16. detectable_format:json_format - Verifies that the entire response is in valid JSON format.
    - (No additional arguments required)

17. detectable_format:title - Checks if the response includes a title wrapped in double angular brackets.
    - (No additional arguments required)

18. combination:two_responses - Ensures that two distinct responses are provided, separated by asterisks.
    - (No additional arguments required)

19. combination:repeat_prompt - Verifies that the original prompt is repeated before the answer is given.
    - prompt_to_repeat: str

20. startend:end_checker - Checks if the response ends with a specific phrase.
    - end_phrase: str

21. change_case:capital_word_frequency - Verifies that a certain number of words are in all capital letters.
    - capital_frequency: int
    - capital_relation: str

22. change_case:english_capital - Ensures the entire response is in English and uses all capital letters.
    - (No additional arguments required)

23. change_case:english_lowercase - Checks that the entire response is in English and uses all lowercase letters.
    - (No additional arguments required)

24. punctuation:no_comma - Verifies that the response does not contain any commas.
    - (No additional arguments required)

25. startend:quotation - Ensures the entire response is wrapped in double quotation marks.
    - (No additional arguments required)

All the arguments are optional.

## Instructions

1. Analyze the provided list of constraints and the given instruction carefully.
2. For each constraint in the list, identify the relevant parameters from the instruction text.
3. If a constraint type is not in the list of constraints then all its arguments should be `null`.
4. Use appropriate data types for the kwargs (strings, integers, booleans, lists, etc.).
5. If an argument is not relevant, then its value must be `null`.
6. Be precise and avoid adding unnecessary or speculative kwargs.
7. For `*_relation` arguments, only provide a non-null value if the corresponding main argument is also non-null.

## Output format:

```
{{
    "keywords": [...],
    "keyword": ...,
    "frequency": ...,
    "relation": ...,
    "forbidden_words": [...],
    "letter": ...,
    "let_frequency": ...,
    "let_relation": ...,
    "language": "...",
    "num_sentences": ...,
    "num_paragraphs": ...,
    "num_words": ...,
    "nth_paragraph": ...,
    "first_word": ...,
    "num_placeholders": ...,
    "postscript_marker": ...,
    "num_bullets": ...,
    "num_highlights": ...,
    "section_spliter": ...,
    "num_sections": ...,
    "prompt_to_repeat": ...,
    "end_phrase": ...,
    "capital_frequency": ...,
    "capital_relation": ...
}}
```
""".lstrip()
