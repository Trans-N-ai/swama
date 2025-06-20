import ArgumentParser
import Foundation
import SwamaKit

// MARK: - List

struct List: AsyncParsableCommand {
    // MARK: Internal

    enum OutputFormat: String, ExpressibleByArgument {
        case text
        case json
    }

    static let configuration: CommandConfiguration = .init(
        commandName: "list",
        abstract: "List all available MLX models."
    )

    @Option(name: .shortAndLong, help: "Format for the output: text, json.")
    var format: OutputFormat = .text

    func run() async throws {
        let models = ModelManager.models()

        if models.isEmpty {
            print("No models found.")
            return
        }

        switch format {
        case .text:
            printModelsText(models)
        case .json:
            try printModelsJSON(models)
        }
    }

    // MARK: Private

    private func printModelsText(_ models: [ModelInfo]) {
        // Determine column widths
        let nameHeader = "NAME"
        let sizeHeader = "SIZE"
        let modifiedHeader = "MODIFIED"

        var maxNameLen = nameHeader.count
        var maxSizeLen = sizeHeader.count
        var maxModifiedLen = modifiedHeader.count

        let formattedModels = models.map { model -> (name: String, id: String, size: String, modified: String) in
            let name = model.id
            let sizeInMB = Double(model.sizeInBytes) / (1024.0 * 1024.0)
            let size =
                if sizeInMB >= 1024.0 {
                    String(format: "%.1f GB", sizeInMB / 1024.0)
                }
                else {
                    String(format: "%.1f MB", sizeInMB)
                }

            let modified = formatTimeAgo(timestamp: TimeInterval(model.created))

            maxNameLen = Swift.max(maxNameLen, name.count)
            maxSizeLen = Swift.max(maxSizeLen, size.count)
            maxModifiedLen = Swift.max(maxModifiedLen, modified.count)

            return (name: name, id: model.id, size: size, modified: modified)
        }

        // Header with NAME, ID (short hash for uniqueness if needed, or full ID if short), SIZE, MODIFIED
        // For now, let's use full ID for NAME, and a placeholder for a shorter ID if we decide to implement it.
        // We will display the full model.id as NAME, and can add a short ID column if needed later.
        // The request was to be like ollama, which uses NAME, ID (short hash), SIZE, MODIFIED.
        // Let's simplify for now and use NAME (full model id), SIZE, MODIFIED.
        print(
            "\(nameHeader.padding(toLength: maxNameLen, withPad: " ", startingAt: 0))  \(sizeHeader.padding(toLength: maxSizeLen, withPad: " ", startingAt: 0))  \(modifiedHeader.padding(toLength: maxModifiedLen, withPad: " ", startingAt: 0))"
        )

        // Print model rows
        for fm in formattedModels {
            let namePart = fm.name.padding(toLength: maxNameLen, withPad: " ", startingAt: 0)
            let sizePart = fm.size.padding(toLength: maxSizeLen, withPad: " ", startingAt: 0)
            let modifiedPart = fm.modified.padding(toLength: maxModifiedLen, withPad: " ", startingAt: 0)
            print("\(namePart)  \(sizePart)  \(modifiedPart)")
        }
    }

    private func formatTimeAgo(timestamp: TimeInterval) -> String {
        let now = Date().timeIntervalSince1970
        let secondsAgo = now - timestamp

        if secondsAgo < 0 {
            return "in the future"
        } // Should not happen
        if secondsAgo < 60 {
            return "just now"
        }

        let minutesAgo = Int(secondsAgo / 60)
        if minutesAgo < 60 {
            return "\(minutesAgo) minute\(minutesAgo == 1 ? "" : "s") ago"
        }

        let hoursAgo = Int(minutesAgo / 60)
        if hoursAgo < 24 {
            return "\(hoursAgo) hour\(hoursAgo == 1 ? "" : "s") ago"
        }

        let daysAgo = Int(hoursAgo / 24)
        if daysAgo < 30 {
            if daysAgo == 1 {
                return "yesterday"
            }
            return "\(daysAgo) day\(daysAgo == 1 ? "" : "s") ago"
        }

        let monthsAgo = Int(daysAgo / 30) // Approximate
        if monthsAgo < 12 {
            return "\(monthsAgo) month\(monthsAgo == 1 ? "" : "s") ago"
        }

        let yearsAgo = Int(monthsAgo / 12)
        return "\(yearsAgo) year\(yearsAgo == 1 ? "" : "s") ago"
    }

    private func printModelsJSON(_ models: [ModelInfo]) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted

        // Prepare a serializable version of ModelInfo
        let serializableModels = models.map { modelInfo -> [String: AnyCodable] in
            var dict: [String: AnyCodable] = [
                "id": AnyCodable(modelInfo.id),
                "created": AnyCodable(modelInfo.created),
                "size_in_bytes": AnyCodable(modelInfo.sizeInBytes),
                "source": AnyCodable(modelInfo.source == .metaFile ? "metaFile" : "directoryScan")
            ]
            if let rawMetadata = modelInfo.rawMetadata {
                dict["raw_metadata"] = wrapAnyCodable(rawMetadata)
            }
            return dict
        }

        let data = try encoder.encode(serializableModels)
        if let jsonString = String(data: data, encoding: .utf8) {
            print(jsonString)
        }
    }

    private func wrapAnyCodable(_ value: Any) -> AnyCodable {
        if let dict = value as? [String: Any] {
            let wrappedDict = dict.mapValues { wrapAnyCodable($0) }
            return AnyCodable(wrappedDict)
        }
        else if let array = value as? [Any] {
            let wrappedArray = array.map { wrapAnyCodable($0) }
            return AnyCodable(wrappedArray)
        }
        else if isBasicType(value) {
            return AnyCodable(value)
        }
        else {
            // Convert unsupported types to string description
            return AnyCodable(String(describing: value))
        }
    }

    private func isBasicType(_ value: Any) -> Bool {
        value is Int ||
            value is Int64 ||
            value is UInt ||
            value is String ||
            value is Double ||
            value is Float ||
            value is Bool ||
            value is NSNull
    }
}

// MARK: - AnyCodable

/// Helper to make [String: Any] encodable
struct AnyCodable: Encodable {
    // MARK: Lifecycle

    init(_ value: Any) {
        self.value = value
    }

    // MARK: Internal

    let value: Any

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let intValue = value as? Int {
            try container.encode(intValue)
        }
        else if let stringValue = value as? String {
            try container.encode(stringValue)
        }
        else if let doubleValue = value as? Double {
            try container.encode(doubleValue)
        }
        else if let boolValue = value as? Bool {
            try container.encode(boolValue)
        }
        else if let arrayValue = value as? [AnyCodable] {
            try container.encode(arrayValue)
        }
        else if let dictionaryValue = value as? [String: AnyCodable] {
            try container.encode(dictionaryValue)
        }
        else if let int64Value = value as? Int64 {
            try container.encode(int64Value)
        }
        else if let uintValue = value as? UInt {
            try container.encode(uintValue)
        }
        else if let floatValue = value as? Float {
            try container.encode(floatValue)
        }
        else {
            // Convert unsupported types to string description
            try container.encode(String(describing: value))
        }
    }
}
