import Testing
import Foundation
@testable import OpenFoundationModelsOpenAI

@Suite("Custom Model Tests")
struct CustomModelTests: Sendable {
    
    @Test("Custom model can be created with configurable properties")
    func testCustomModelCreation() {
        let customConfig = OpenAIModel.CustomModelConfig(
            contextWindow: 100_000,
            maxOutputTokens: 8_192,
            capabilities: [.textGeneration, .functionCalling, .streaming],
            pricingTier: .standard,
            knowledgeCutoff: "January 2025",
            modelType: .gpt,
            parameterConstraints: ParameterConstraints(
                supportsTemperature: true,
                supportsTopP: true,
                supportsFrequencyPenalty: true,
                supportsPresencePenalty: true,
                supportsStop: true,
                maxTokensParameterName: "max_tokens",
                temperatureRange: 0...1.0,
                topPRange: 0.0...1.0
            )
        )
        
        let customModel = OpenAIModel.custom(name: "my-custom-model", config: customConfig)
        
        #expect(customModel.apiName == "my-custom-model")
        #expect(customModel.contextWindow == 100_000)
        #expect(customModel.maxOutputTokens == 8_192)
        #expect(customModel.pricingTier == .standard)
        #expect(customModel.knowledgeCutoff == "January 2025")
        #expect(customModel.supportsFunctionCalling)
        #expect(customModel.supportsStreaming)
        #expect(!customModel.supportsVision)
        #expect(!customModel.isReasoningModel)
    }
    
    @Test("Custom model with reasoning capabilities")
    func testCustomReasoningModel() {
        let customConfig = OpenAIModel.CustomModelConfig(
            contextWindow: 200_000,
            maxOutputTokens: 32_768,
            capabilities: [.textGeneration, .reasoning, .functionCalling, .streaming, .toolAccess],
            pricingTier: .premium,
            knowledgeCutoff: "June 2024",
            modelType: .reasoning,
            parameterConstraints: ParameterConstraints(
                supportsTemperature: false,
                supportsTopP: false,
                supportsFrequencyPenalty: false,
                supportsPresencePenalty: false,
                supportsStop: false,
                maxTokensParameterName: "max_completion_tokens",
                temperatureRange: nil,
                topPRange: nil
            )
        )
        
        let customModel = OpenAIModel.custom(name: "custom-reasoning-model", config: customConfig)
        
        #expect(customModel.apiName == "custom-reasoning-model")
        #expect(customModel.contextWindow == 200_000)
        #expect(customModel.maxOutputTokens == 32_768)
        #expect(customModel.pricingTier == .premium)
        #expect(customModel.knowledgeCutoff == "June 2024")
        #expect(customModel.isReasoningModel)
        #expect(customModel.supportsFunctionCalling)
        #expect(customModel.supportsStreaming)
        #expect(customModel.capabilities.contains(.reasoning))
        #expect(!customModel.supportsVision)
    }
    
    @Test("Predefined models still work as before")
    func testPredefinedModelsStillWork() {
        let gpt4o = OpenAIModel.gpt4o
        let o1 = OpenAIModel.o1
        
        #expect(gpt4o.apiName == "gpt-4o")
        #expect(gpt4o.contextWindow == 128_000)
        #expect(gpt4o.supportsVision)
        #expect(!gpt4o.isReasoningModel)
        
        #expect(o1.apiName == "o1")
        #expect(o1.contextWindow == 200_000)
        #expect(o1.isReasoningModel)
        #expect(!o1.supportsVision)
    }
    
    @Test("Custom model equality comparison")
    func testCustomModelEquality() {
        let config1 = OpenAIModel.CustomModelConfig(
            contextWindow: 100_000,
            maxOutputTokens: 8_192,
            capabilities: [.textGeneration, .functionCalling],
            pricingTier: .standard,
            knowledgeCutoff: "January 2025",
            modelType: .gpt,
            parameterConstraints: ParameterConstraints(
                supportsTemperature: true,
                supportsTopP: true,
                supportsFrequencyPenalty: true,
                supportsPresencePenalty: true,
                supportsStop: true,
                maxTokensParameterName: "max_tokens",
                temperatureRange: 0.0...1.0,
                topPRange: 0.0...1.0
            )
        )
        
        let config2 = OpenAIModel.CustomModelConfig(
            contextWindow: 100_000,
            maxOutputTokens: 8_192,
            capabilities: [.textGeneration, .functionCalling],
            pricingTier: .standard,
            knowledgeCutoff: "January 2025",
            modelType: .gpt,
            parameterConstraints: ParameterConstraints(
                supportsTemperature: true,
                supportsTopP: true,
                supportsFrequencyPenalty: true,
                supportsPresencePenalty: true,
                supportsStop: true,
                maxTokensParameterName: "max_tokens",
                temperatureRange: 0.0...1.0,
                topPRange: 0.0...1.0
            )
        )
        
        let customModel1 = OpenAIModel.custom(name: "test-model", config: config1)
        let customModel2 = OpenAIModel.custom(name: "test-model", config: config2)
        let customModel3 = OpenAIModel.custom(name: "different-model", config: config1)
        
        #expect(customModel1 == customModel2)
        #expect(customModel1 != customModel3)
    }
}