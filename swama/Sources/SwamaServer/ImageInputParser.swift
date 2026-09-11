import Foundation
import SwamaCore

/// The single image-input validator shared by every HTTP adapter.
///
/// Both `/v1/chat/completions` and `/v1/responses` accept user-supplied image
/// references, so the rules live here exactly once. Two copies of a
/// security-sensitive parser drift: the Responses adapter previously carried a
/// weaker scheme-only check and accepted inputs Chat already rejected.
///
/// Returns `nil` for anything unsupported; each adapter maps that to its own
/// error type so their public contracts stay unchanged.
enum ImageInputParser {
    static func contentPart(_ value: String) -> SwamaCore.ContentPart? {
        if value.lowercased().hasPrefix("data:") {
            return dataImagePart(value)
        }

        guard let url = URL(string: value, encodingInvalidCharacters: false),
              let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let scheme = components.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              components.host?.isEmpty == false,
              components.user == nil,
              components.password == nil,
              hasValidExplicitPort(in: value),
              components.port.map({ (1 ... 65535).contains($0) }) ?? true
        else {
            return nil
        }

        return .imageURL(url)
    }

    private static func dataImagePart(_ value: String) -> SwamaCore.ContentPart? {
        guard let comma = value.firstIndex(of: ",") else {
            return nil
        }

        let header = String(value[..<comma])
        let encoded = String(value[value.index(after: comma)...])
        let components = header.split(separator: ";", omittingEmptySubsequences: false)
        guard components.count == 2,
              let mediaSubtype = components[0].split(separator: "/", omittingEmptySubsequences: false).last,
              components[0].lowercased().hasPrefix("data:image/"),
              components[0].count(where: { $0 == "/" }) == 1,
              isASCIIMediaSubtype(mediaSubtype),
              components[1].lowercased() == "base64",
              let data = Data(base64Encoded: encoded),
              !data.isEmpty
        else {
            return nil
        }

        return .imageData(data, mediaType: String(components[0].dropFirst("data:".count)))
    }

    private static func hasValidExplicitPort(in value: String) -> Bool {
        guard let schemeSeparator = value.range(of: "://") else {
            return false
        }

        let authorityStart = schemeSeparator.upperBound
        let authorityEnd = value[authorityStart...].firstIndex { ["/", "?", "#"].contains($0) } ?? value.endIndex
        let authority = value[authorityStart ..< authorityEnd]

        if authority.first == "[" {
            guard let closingBracket = authority.firstIndex(of: "]") else {
                return false
            }

            let remainder = authority[authority.index(after: closingBracket)...]
            guard !remainder.isEmpty else {
                return true
            }
            guard remainder.first == ":" else {
                return false
            }

            return isValidPort(remainder.dropFirst())
        }

        guard let separator = authority.lastIndex(of: ":") else {
            return true
        }

        return isValidPort(authority[authority.index(after: separator)...])
    }

    private static func isValidPort(_ value: Substring) -> Bool {
        guard !value.isEmpty,
              value.utf8.allSatisfy({ (48 ... 57).contains($0) }),
              let port = Int(value)
        else {
            return false
        }

        return (1 ... 65535).contains(port)
    }

    private static func isASCIIMediaSubtype(_ value: Substring) -> Bool {
        guard !value.isEmpty else {
            return false
        }

        return value.utf8.allSatisfy { byte in
            switch byte {
            case 33,
                 35 ... 39,
                 42,
                 43,
                 45,
                 46,
                 48 ... 57,
                 65 ... 90,
                 94 ... 96,
                 97 ... 122,
                 124,
                 126:
                true
            default:
                false
            }
        }
    }
}
