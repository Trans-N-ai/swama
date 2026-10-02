import Foundation

/// Ordered JSON for SystemOne maps: criteria order determines the A/B/... labels.
/// Foundation validates the complete document before this traversal records its order.
indirect enum SystemOneJSON: Sendable, Equatable {
    struct Member: Sendable, Equatable {
        let name: String
        let value: SystemOneJSON
    }

    case object([Member])
    case array([SystemOneJSON])
    case string(String)
    case number(String)
    case bool(Bool)
    case null

    static func parse(_ data: Data) throws -> SystemOneJSON {
        _ = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        var cursor = Cursor(bytes: Array(data))
        let value = try cursor.value(depth: 0)
        cursor.whitespace()
        guard cursor.position == cursor.bytes.count else {
            throw DecisionWireError.invalid("Invalid SystemOne JSON document.")
        }

        return value
    }

    func member(_ name: String) -> SystemOneJSON? {
        guard case let .object(members) = self else {
            return nil
        }

        return members.first { $0.name == name }?.value
    }

    var text: String {
        switch self {
        case .null: ""
        case let .string(value): value
        default: json
        }
    }

    /// SGLang render_text: json.dumps(..., ensure_ascii=False, separators=(",", ":")).
    /// Object order and integer-versus-floating syntax are retained from the request.
    var json: String {
        switch self {
        case let .object(members):
            "{" + members.map { Self.quoted($0.name) + ":" + $0.value.json }.joined(separator: ",") + "}"
        case let .array(values): "[" + values.map(\.json).joined(separator: ",") + "]"
        case let .string(value): Self.quoted(value)
        case let .number(raw): Self.normalizedNumber(raw)
        case let .bool(value): value ? "true" : "false"
        case .null: "null"
        }
    }

    static func object(_ pairs: [(String, SystemOneJSON)]) -> SystemOneJSON {
        .object(pairs.map { Member(name: $0.0, value: $0.1) })
    }

    static func number(_ value: Double) -> SystemOneJSON { .number(String(value)) }
    static func integer(_ value: Int) -> SystemOneJSON { .number(String(value)) }

    private static func quoted(_ value: String) -> String {
        var output = "\""
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 0x22: output += "\\\""
            case 0x5C: output += "\\\\"
            case 0x08: output += "\\b"
            case 0x0C: output += "\\f"
            case 0x0A: output += "\\n"
            case 0x0D: output += "\\r"
            case 0x09: output += "\\t"
            case 0 ..< 0x20: output += String(format: "\\u%04x", scalar.value)
            default: output.unicodeScalars.append(scalar)
            }
        }
        return output + "\""
    }

    private static func normalizedNumber(_ raw: String) -> String {
        if raw.contains(".") || raw.contains("e") || raw.contains("E") {
            // The parser checks that floating values are finite. Double.description uses
            // locale-independent round-trip text and retains .0 on integral doubles.
            return String(Double(raw)!)
        }
        return raw == "-0" ? "0" : raw
    }

    private struct Cursor {
        let bytes: [UInt8]
        var position = 0

        mutating func whitespace() {
            while position < bytes.count, [9, 10, 13, 32].contains(bytes[position]) {
                position += 1
            }
        }

        mutating func consume(_ byte: UInt8) throws {
            whitespace()
            guard position < bytes.count, bytes[position] == byte else {
                throw DecisionWireError.invalid("Invalid SystemOne JSON document.")
            }

            position += 1
        }

        mutating func value(depth: Int) throws -> SystemOneJSON {
            guard depth <= 256 else {
                throw DecisionWireError.invalid("SystemOne JSON is nested too deeply.")
            }

            whitespace()
            guard position < bytes.count else {
                throw DecisionWireError.invalid("Invalid SystemOne JSON document.")
            }

            switch bytes[position] {
            case 123:
                position += 1
                var members = [Member](), names = Set<Data>()
                whitespace()
                if position < bytes.count, bytes[position] == 125 {
                    position += 1; return .object(members)
                }
                while true {
                    let name = try string()
                    guard names.insert(Data(name.utf8)).inserted else {
                        throw DecisionWireError.invalid("Duplicate SystemOne object key '\(name)'.")
                    }

                    try consume(58)
                    try members.append(.init(name: name, value: value(depth: depth + 1)))
                    whitespace()
                    if position < bytes.count, bytes[position] == 125 {
                        position += 1; break
                    }
                    try consume(44)
                }
                return .object(members)

            case 91:
                position += 1
                var values = [SystemOneJSON]()
                whitespace()
                if position < bytes.count, bytes[position] == 93 {
                    position += 1; return .array(values)
                }
                while true {
                    try values.append(value(depth: depth + 1))
                    whitespace()
                    if position < bytes.count, bytes[position] == 93 {
                        position += 1; break
                    }
                    try consume(44)
                }
                return .array(values)

            case 34: return try .string(string())

            case 116: position += 4; return .bool(true)

            case 102: position += 5; return .bool(false)

            case 110: position += 4; return .null

            default:
                let start = position
                while position < bytes.count, ![9, 10, 13, 32, 44, 93, 125].contains(bytes[position]) {
                    position += 1
                }
                let raw = String(decoding: bytes[start ..< position], as: UTF8.self)
                guard !raw.isEmpty else {
                    throw DecisionWireError.invalid("Invalid SystemOne JSON number.")
                }

                if raw.contains(".") || raw.contains("e") || raw.contains("E") {
                    guard let number = Double(raw), number.isFinite else {
                        throw DecisionWireError.invalid("SystemOne JSON numbers must be finite.")
                    }
                }
                return .number(raw)
            }
        }

        mutating func string() throws -> String {
            whitespace()
            let start = position
            try consume(34)
            while position < bytes.count {
                if bytes[position] == 92 {
                    position += 2; continue
                }
                if bytes[position] == 34 {
                    position += 1
                    return try JSONDecoder().decode(String.self, from: Data(bytes[start ..< position]))
                }
                position += 1
            }
            throw DecisionWireError.invalid("Invalid SystemOne JSON string.")
        }
    }
}
