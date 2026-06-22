Hey. This is a simple ai agent/chatbot. It uses the deepseek api, so the format will work with any deepseek model. Set your api key as an env. variable.

This is an agent with one simple twist. Every line of text from the agent or the user is stored in a global kv cache. Tool calls and tool results are not stored. All conversational text is just
stored flatly in the file, no session history etc... Super simple.

The agent as a tool called 'recall', it can search the global cache if needed. Semantic search is enabled by default. i.e: If the agent searches for 'writers' you might like, 'authors' will come up as well.

The entire kv cache is never exposed to the agent in order to manage context, just the results of the search. This allows it to 'remember' every conversation you ever had, across any 'session' without bloated context.

No web search.

Works on my machine, macOS, have not tested on others. This is just a personal project.
Claude helped me with this one

Dependencies in the project.toml
HF dependency is for the semantic search, its not necessary but strongly recommended.

Thanks!
Faisal
