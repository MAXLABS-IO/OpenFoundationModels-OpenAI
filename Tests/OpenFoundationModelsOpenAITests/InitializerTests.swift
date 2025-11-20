import Testing
import Foundation
@testable import OpenFoundationModelsOpenAI

@Suite("Public Initializer Tests")
struct PublicInitializerTests: Sendable {
    
    @Test("CustomModelConfig can be initialized publicly")
    func testCustomModelConfigPublicInitializer() {
        let constraints = ParameterConstraints(
            supportsTemperature: true,
            supportsTopP: true,
            supportsFrequencyPenalty: false,
            supportsPresencePenalty: false,
            supportsStop: true,
            maxTokensParameterName: "max_tokens",
            temperatureRange: 0.0...1.0,
            topPRange: 0.0...1.0
        )
        
        let config = OpenAIModel.CustomModelConfig(
            contextWindow: 100_000,
            maxOutputTokens: 8_192,
            capabilities: [.textGeneration, .functionCalling],
            pricingTier: .standard,
            knowledgeCutoff: "January 2025",
            modelType: .gpt,
            parameterConstraints: constraints
        )
        
        #expect(config.contextWindow == 100_000)
        #expect(config.maxOutputTokens == 8_192)
        #expect(config.pricingTier == .standard)
        #expect(config.knowledgeCutoff == "January 2025")
        #expect(config.modelType == .gpt)
        #expect(config.capabilities.contains(.textGeneration))
        #expect(config.parameterConstraints?.supportsTemperature == true)
        #expect(config.parameterConstraints?.supportsTopP == true)
        #expect(config.parameterConstraints?.supportsStop == true)
    }
    
    @Test("ParameterConstraints can be initialized publicly")
    func testParameterConstraintsPublicInitializer() {
        let constraints = ParameterConstraints(
            supportsTemperature: true,
            supportsTopP: false,
            supportsFrequencyPenalty: true,
            supportsPresencePenalty: false,
            supportsStop: true,
            maxTokensParameterName: "max_completion_tokens",
            temperatureRange: 0.0...2.0,
            topPRange: nil
        )
        
        #expect(constraints.supportsTemperature == true)
        #expect(constraints.supportsTopP == false)
        #expect(constraints.supportsFrequencyPenalty == true)
        #expect(constraints.supportsPresencePenalty == false)
        #expect(constraints.supportsStop == true)
        #expect(constraints.maxTokensParameterName == "max_completion_tokens")
        #expect(constraints.temperatureRange == 0...2.0)
        #expect(constraints.topPRange == nil)
    }
    
    @Test("ModelType can be used publicly")
    func testModelTypePublicUsage() {
        let gptType: ModelType = .gpt
        let reasoningType: ModelType = .reasoning
        
        #expect(gptType == .gpt)
        #expect(reasoningType == .reasoning)
        #expect(gptType != reasoningType)
    }
    
    @Test("Complete custom model creation flow with explicit constraints")
    func testCompleteCustomModelFlowWithExplicitConstraints() {
        // Create parameter constraints
        let constraints = ParameterConstraints(
            supportsTemperature: true,
            supportsTopP: true,
            supportsFrequencyPenalty: true,
            supportsPresencePenalty: true,
            supportsStop: true,
            maxTokensParameterName: "max_tokens",
            temperatureRange: 0.0...1.0,
            topPRange: 0.0...1.0
        )
        
        // Create model config
        let config = OpenAIModel.CustomModelConfig(
            contextWindow: 50_000,
            maxOutputTokens: 4_096,
            capabilities: [.textGeneration, .functionCalling, .streaming, .vision],
            pricingTier: .economy,
            knowledgeCutoff: "December 2024",
            modelType: .gpt,
            parameterConstraints: constraints
        )
        
        // Create custom model
        let customModel = OpenAIModel.custom(name: "test-custom-model", config: config)
        
        // Verify the model properties
        #expect(customModel.apiName == "test-custom-model")
        #expect(customModel.contextWindow == 50_000)
        #expect(customModel.maxOutputTokens == 4_096)
        #expect(customModel.pricingTier == .economy)
        #expect(customModel.knowledgeCutoff == "December 2024")
        #expect(customModel.modelType == .gpt)
        #expect(customModel.supportsVision)
        #expect(customModel.supportsFunctionCalling)
        #expect(customModel.supportsStreaming)
        #expect(!customModel.isReasoningModel)
        #expect(customModel.constraints.supportsTemperature)
        #expect(customModel.constraints.supportsTopP)
    }
    
    @Test("Custom model with default constraints when nil is provided")
    func testCustomModelWithDefaultConstraints() {
        // Create model config with nil constraints (should use defaults)
        let config = OpenAIModel.CustomModelConfig(
            contextWindow: 50_000,
            maxOutputTokens: 4_096,
            capabilities: [.textGeneration, .functionCalling, .streaming, .vision],
            pricingTier: .economy,
            knowledgeCutoff: "December 2024",
            modelType: .gpt,
            parameterConstraints: nil  // Use default constraints
        )
        
        // Create custom model
        let customModel = OpenAIModel.custom(name: "test-custom-model-default", config: config)
        
        // Verify that default GPT constraints are applied
        #expect(customModel.constraints.supportsTemperature == true)
        #expect(customModel.constraints.supportsTopP == true)
        #expect(customModel.constraints.supportsFrequencyPenalty == true)
        #expect(customModel.constraints.supportsPresencePenalty == true)
        #expect(customModel.constraints.supportsStop == true)
        #expect(customModel.constraints.maxTokensParameterName == "max_tokens")
        #expect(customModel.constraints.temperatureRange == 0.0...2.0)
        #expect(customModel.constraints.topPRange == 0...1.0)
    }
    
    @Test("Custom reasoning model with default constraints when nil is provided")
    func testCustomReasoningModelWithDefaultConstraints() {
        // Create model config with nil constraints (should use defaults for reasoning)
        let config = OpenAIModel.CustomModelConfig(
            contextWindow: 100_000,
            maxOutputTokens: 32_768,
            capabilities: [.textGeneration, .reasoning, .functionCalling],
            pricingTier: .premium,
            knowledgeCutoff: "December 2024",
            modelType: .reasoning,
            parameterConstraints: nil  // Use default constraints
        )
        
        // Create custom model
        let customModel = OpenAIModel.custom(name: "test-reasoning-model-default", config: config)
        
        // Verify that default reasoning constraints are applied
        #expect(customModel.constraints.supportsTemperature == false)
        #expect(customModel.constraints.supportsTopP == false)
        #expect(customModel.constraints.supportsFrequencyPenalty == false)
        #expect(customModel.constraints.supportsPresencePenalty == false)
        #expect(customModel.constraints.supportsStop == false)
        #expect(customModel.constraints.maxTokensParameterName == "max_completion_tokens")
        #expect(customModel.constraints.temperatureRange == nil)
        #expect(customModel.constraints.topPRange == nil)
    }
}