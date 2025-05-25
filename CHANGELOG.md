# Changelog

## v0.3.0

- feat: new `before_call` and `after_call` callbacks to help users provide hooks before and after tool calls (used for HITL as well)
- feat: `AgentTeam` and `AgentPipeline` instances can now be converted into ASGI APIs
- feat: Enhanced `Context` objects to provide clear, resumable and easy way to provide/retrieve/resume Agent calls.
- feat: Enhanced `Step` objects to provide clear, resumable and easy way to provide/retrieve/resume Agent calls.
- feat: Tool calls and structured outputs in `CerebrasGenerationProvider`
- feat: new ModelKind type variable to help users to obtain provider-specific models for specific categories, like "standard", "nano", "mini", etc. Useful when using `FailoverGenerationProvider` since the model you might use in one might not be present in the other, but you want the same kind of model for each provider you are using.
- feat: new `timeout_m`, `timeout_s` to generation configuration
- fix: wrong pricing calculations in `GoogleGenerationProvider` and `CerebrasGenerationProvider` classes.
- fix: prices not showing up properly in Langfuse view.
