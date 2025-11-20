import Foundation

// MARK: - Unified OpenAI Model
public struct OpenAIModel: Sendable, Hashable {
    // Internal enum for predefined models (maintains CaseIterable)
    internal enum PredefinedModel: String, CaseIterable, Sendable {
        // GPT Family Models
        case gpt4o = "gpt-4o"
        case gpt4oMini = "gpt-4o-mini"
        case gpt4Turbo = "gpt-4-turbo"
        
        // Reasoning Family Models (o-series)
        case o1 = "o1"
        case o1Pro = "o1-pro"
        case o3 = "o3"
        case o3Pro = "o3-pro"
        case o4Mini = "o4-mini"
    }
    
    // Model identifier to distinguish between predefined and custom models
    private var modelIdentifier: ModelIdentifier
    
    private enum ModelIdentifier: Hashable {
        case predefined(PredefinedModel)
        case custom(name: String, config: CustomModelConfig)
    }
    
    // Custom model configuration
    public struct CustomModelConfig: Sendable, Hashable {
        public let contextWindow: Int
        public let maxOutputTokens: Int
        public let capabilities: ModelCapabilities
        public let pricingTier: PricingTier
        public let knowledgeCutoff: String
        public let modelType: ModelType
        public let parameterConstraints: ParameterConstraints?
        
        public init(
            contextWindow: Int,
            maxOutputTokens: Int,
            capabilities: ModelCapabilities,
            pricingTier: PricingTier,
            knowledgeCutoff: String,
            modelType: ModelType,
            parameterConstraints: ParameterConstraints? = nil
        ) {
            self.contextWindow = contextWindow
            self.maxOutputTokens = maxOutputTokens
            self.capabilities = capabilities
            self.pricingTier = pricingTier
            self.knowledgeCutoff = knowledgeCutoff
            self.modelType = modelType
            self.parameterConstraints = parameterConstraints
        }
        
        public static func == (lhs: CustomModelConfig, rhs: CustomModelConfig) -> Bool {
            return lhs.contextWindow == rhs.contextWindow &&
                   lhs.maxOutputTokens == rhs.maxOutputTokens &&
                   lhs.capabilities == rhs.capabilities &&
                   lhs.pricingTier == rhs.pricingTier &&
                   lhs.knowledgeCutoff == rhs.knowledgeCutoff &&
                   lhs.modelType == rhs.modelType &&
                   lhs.parameterConstraints?.supportsTemperature == rhs.parameterConstraints?.supportsTemperature &&
                   lhs.parameterConstraints?.supportsTopP == rhs.parameterConstraints?.supportsTopP &&
                   lhs.parameterConstraints?.supportsFrequencyPenalty == rhs.parameterConstraints?.supportsFrequencyPenalty &&
                   lhs.parameterConstraints?.supportsPresencePenalty == rhs.parameterConstraints?.supportsPresencePenalty &&
                   lhs.parameterConstraints?.supportsStop == rhs.parameterConstraints?.supportsStop &&
                   lhs.parameterConstraints?.maxTokensParameterName == rhs.parameterConstraints?.maxTokensParameterName &&
                   lhs.parameterConstraints?.temperatureRange == rhs.parameterConstraints?.temperatureRange &&
                   lhs.parameterConstraints?.topPRange == rhs.parameterConstraints?.topPRange
        }
        
        public func hash(into hasher: inout Hasher) {
            hasher.combine(contextWindow)
            hasher.combine(maxOutputTokens)
            hasher.combine(capabilities)
            hasher.combine(pricingTier)
            hasher.combine(knowledgeCutoff)
            hasher.combine(modelType)
            hasher.combine(parameterConstraints?.supportsTemperature)
            hasher.combine(parameterConstraints?.supportsTopP)
            hasher.combine(parameterConstraints?.supportsFrequencyPenalty)
            hasher.combine(parameterConstraints?.supportsPresencePenalty)
            hasher.combine(parameterConstraints?.supportsStop)
            hasher.combine(parameterConstraints?.maxTokensParameterName)
            hasher.combine(parameterConstraints?.temperatureRange)
            hasher.combine(parameterConstraints?.topPRange)
        }
    }
    
    // Private initializer
    private init(_ identifier: ModelIdentifier) {
        self.modelIdentifier = identifier
    }
    
    // Predefined model initializers (maintain existing API)
    public static let gpt4o = OpenAIModel(.predefined(.gpt4o))
    public static let gpt4oMini = OpenAIModel(.predefined(.gpt4oMini))
    public static let gpt4Turbo = OpenAIModel(.predefined(.gpt4Turbo))
    public static let o1 = OpenAIModel(.predefined(.o1))
    public static let o1Pro = OpenAIModel(.predefined(.o1Pro))
    public static let o3 = OpenAIModel(.predefined(.o3))
    public static let o3Pro = OpenAIModel(.predefined(.o3Pro))
    public static let o4Mini = OpenAIModel(.predefined(.o4Mini))
    
    // Custom model initializer
    public static func custom(name: String, config: CustomModelConfig) -> OpenAIModel {
        return OpenAIModel(.custom(name: name, config: config))
    }
    
    // MARK: - Model Properties
    
    /// API name used in requests
    public var apiName: String {
        switch modelIdentifier {
        case .predefined(let predefined):
            return predefined.rawValue
        case .custom(let name, _):
            return name
        }
    }
    
    /// Context window size in tokens
    public var contextWindow: Int {
        switch modelIdentifier {
        case .predefined(let model):
            switch model {
            case .gpt4o, .gpt4oMini, .gpt4Turbo:
                return 128_000
            case .o1, .o1Pro, .o3, .o3Pro, .o4Mini:
                return 200_000 // Reasoning models typically have larger context
            }
        case .custom(_, let config):
            return config.contextWindow
        }
    }
    
    /// Maximum output tokens
    public var maxOutputTokens: Int {
        switch modelIdentifier {
        case .predefined(let model):
            switch model {
            case .gpt4o, .gpt4oMini:
                return 16_384
            case .gpt4Turbo:
                return 4_096
            case .o1, .o3:
                return 32_768
            case .o1Pro, .o3Pro:
                return 65_536
            case .o4Mini:
                return 16_384
            }
        case .custom(_, let config):
            return config.maxOutputTokens
        }
    }
    
    /// Model capabilities
    public var capabilities: ModelCapabilities {
        switch modelIdentifier {
        case .predefined(let model):
            switch modelType {
            case .gpt:
                switch model {
                case .gpt4o, .gpt4Turbo:
                    return [.textGeneration, .vision, .functionCalling, .streaming, .toolAccess]
                case .gpt4oMini:
                    return [.textGeneration, .vision, .functionCalling, .streaming]
                default:
                    return [.textGeneration, .functionCalling, .streaming]
                }
            case .reasoning:
                return [.textGeneration, .reasoning, .functionCalling, .streaming, .toolAccess]
            }
        case .custom(_, let config):
            return config.capabilities
        }
    }
    
    /// Pricing tier
    public var pricingTier: PricingTier {
        switch modelIdentifier {
        case .predefined(let model):
            switch model {
            case .gpt4oMini, .o4Mini:
                return .economy
            case .gpt4o, .gpt4Turbo, .o1, .o3:
                return .standard
            case .o1Pro, .o3Pro:
                return .premium
            }
        case .custom(_, let config):
            return config.pricingTier
        }
    }
    
    /// Knowledge cutoff date
    public var knowledgeCutoff: String {
        switch modelIdentifier {
        case .predefined(let model):
            switch model {
            case .gpt4o, .gpt4oMini:
                return "October 2023"
            case .gpt4Turbo:
                return "April 2024"
            case .o1, .o1Pro, .o3, .o3Pro, .o4Mini:
                return "October 2023"
            }
        case .custom(_, let config):
            return config.knowledgeCutoff
        }
    }
    
    // MARK: - Internal Properties
    
    /// Internal model type for behavior switching
    internal var modelType: ModelType {
        switch modelIdentifier {
        case .predefined(let model):
            switch model {
            case .gpt4o, .gpt4oMini, .gpt4Turbo:
                return .gpt
            case .o1, .o1Pro, .o3, .o3Pro, .o4Mini:
                return .reasoning
            }
        case .custom(_, let config):
            return config.modelType
        }
    }
    
    /// Internal parameter constraints
    internal var constraints: ParameterConstraints {
        switch modelIdentifier {
        case .predefined(_):
            return parameterConstraints(for: modelType)
        case .custom(_, let config):
            if let parameterConstraints = config.parameterConstraints {
                return parameterConstraints
            } else {
                return parameterConstraints(for: config.modelType)
            }
        }
    }
    
    private func parameterConstraints(for modelType: ModelType) -> ParameterConstraints {
        switch modelType {
        case .gpt:
            return ParameterConstraints(
                supportsTemperature: true,
                supportsTopP: true,
                supportsFrequencyPenalty: true,
                supportsPresencePenalty: true,
                supportsStop: true,
                maxTokensParameterName: "max_tokens",
                temperatureRange: 0.0...2.0,
                topPRange: 0.0...1.0
            )
        case .reasoning:
            return ParameterConstraints(
                supportsTemperature: false,
                supportsTopP: false,
                supportsFrequencyPenalty: false,
                supportsPresencePenalty: false,
                supportsStop: false,
                maxTokensParameterName: "max_completion_tokens",
                temperatureRange: nil,
                topPRange: nil
            )
        }
    }
    
    /// Check if model supports vision
    public var supportsVision: Bool {
        return capabilities.contains(.vision)
    }
    
    /// Check if model supports function calling
    public var supportsFunctionCalling: Bool {
        return capabilities.contains(.functionCalling)
    }
    
    /// Check if model supports streaming
    public var supportsStreaming: Bool {
        return capabilities.contains(.streaming)
    }
    
    /// Check if model is a reasoning model
    public var isReasoningModel: Bool {
        return capabilities.contains(.reasoning)
    }
}

// MARK: - Supporting Types

/// Model type for implementation switching
public enum ModelType: Sendable, Hashable {
    case gpt
    case reasoning
}

/// Model capabilities using OptionSet
public struct ModelCapabilities: OptionSet, Sendable, Hashable {
    public let rawValue: Int
    
    public init(rawValue: Int) {
        self.rawValue = rawValue
    }
    
    public static let textGeneration = ModelCapabilities(rawValue: 1 << 0)
    public static let vision = ModelCapabilities(rawValue: 1 << 1)
    public static let functionCalling = ModelCapabilities(rawValue: 1 << 2)
    public static let reasoning = ModelCapabilities(rawValue: 1 << 3)
    public static let toolAccess = ModelCapabilities(rawValue: 1 << 4)
    public static let streaming = ModelCapabilities(rawValue: 1 << 5)
}

/// Parameter constraints for different model types
public struct ParameterConstraints: Sendable, Hashable {
    public let supportsTemperature: Bool
    public let supportsTopP: Bool
    public let supportsFrequencyPenalty: Bool
    public let supportsPresencePenalty: Bool
    public let supportsStop: Bool
    public let maxTokensParameterName: String
    public let temperatureRange: ClosedRange<Double>?
    public let topPRange: ClosedRange<Double>?
    
    public init(
        supportsTemperature: Bool,
        supportsTopP: Bool,
        supportsFrequencyPenalty: Bool,
        supportsPresencePenalty: Bool,
        supportsStop: Bool,
        maxTokensParameterName: String,
        temperatureRange: ClosedRange<Double>?,
        topPRange: ClosedRange<Double>?
    ) {
        self.supportsTemperature = supportsTemperature
        self.supportsTopP = supportsTopP
        self.supportsFrequencyPenalty = supportsFrequencyPenalty
        self.supportsPresencePenalty = supportsPresencePenalty
        self.supportsStop = supportsStop
        self.maxTokensParameterName = maxTokensParameterName
        self.temperatureRange = temperatureRange
        self.topPRange = topPRange
    }
    
    public static func == (lhs: ParameterConstraints, rhs: ParameterConstraints) -> Bool {
        return lhs.supportsTemperature == rhs.supportsTemperature &&
               lhs.supportsTopP == rhs.supportsTopP &&
               lhs.supportsFrequencyPenalty == rhs.supportsFrequencyPenalty &&
               lhs.supportsPresencePenalty == rhs.supportsPresencePenalty &&
               lhs.supportsStop == rhs.supportsStop &&
               lhs.maxTokensParameterName == rhs.maxTokensParameterName &&
               lhs.temperatureRange == rhs.temperatureRange &&
               lhs.topPRange == rhs.topPRange
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(supportsTemperature)
        hasher.combine(supportsTopP)
        hasher.combine(supportsFrequencyPenalty)
        hasher.combine(supportsPresencePenalty)
        hasher.combine(supportsStop)
        hasher.combine(maxTokensParameterName)
        hasher.combine(temperatureRange)
        hasher.combine(topPRange)
    }
}

/// Pricing tiers
public enum PricingTier: String, CaseIterable, Sendable {
    case economy = "economy"
    case standard = "standard"
    case premium = "premium"
    
    public var description: String {
        switch self {
        case .economy:
            return "Cost-efficient models for basic tasks"
        case .standard:
            return "Balanced performance and cost"
        case .premium:
            return "Highest capability models with advanced features"
        }
    }
}

// MARK: - Model Extensions

extension OpenAIModel {

    /// Get all models (equivalent to allPredefinedModels since custom models can't be enumerated)
    public static var allCases: [OpenAIModel] {
        return allPredefinedModels
    }

    /// Get all predefined models
    internal static var allPredefinedModels: [OpenAIModel] {
        return PredefinedModel.allCases.map { OpenAIModel(.predefined($0)) }
    }

    /// Get all models of a specific type
    internal static func models(ofType type: ModelType) -> [OpenAIModel] {
        return allCases.filter { $0.modelType == type }
    }
    
    /// Get all GPT models
    public static var gptModels: [OpenAIModel] {
        return models(ofType: .gpt)
    }
    
    /// Get all reasoning models
    public static var reasoningModels: [OpenAIModel] {
        return models(ofType: .reasoning)
    }
    
    /// Get models by pricing tier
    public static func models(withPricingTier tier: PricingTier) -> [OpenAIModel] {
        return allCases.filter { $0.pricingTier == tier }
    }
    
    /// Get models with specific capability
    public static func models(withCapability capability: ModelCapabilities) -> [OpenAIModel] {
        return allCases.filter { $0.capabilities.contains(capability) }
    }
}

extension OpenAIModel: CustomStringConvertible {
    public var description: String {
        return "\(apiName) (\(modelType), \(pricingTier.rawValue))"
    }
}

extension OpenAIModel: CustomDebugStringConvertible {
    public var debugDescription: String {
        return """
        OpenAIModel(
            name: \(apiName),
            type: \(modelType),
            contextWindow: \(contextWindow),
            maxOutput: \(maxOutputTokens),
            capabilities: \(capabilities),
            pricingTier: \(pricingTier),
            knowledgeCutoff: \(knowledgeCutoff)
        )
        """
    }
}
