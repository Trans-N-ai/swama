import CoreGraphics
import Foundation
import ImageIO
import NIOCore
import NIOEmbedded
import NIOHTTP1
import SwamaCore
@testable import SwamaServer
import Testing
import UniformTypeIdentifiers

// MARK: - SystemOneImagesTests

/// #swama task #82: Cloudflare Clef's `images[]` extension on /v1/systemone, without a model.
@Suite("SystemOne images (Clef extension)")
struct SystemOneImagesTests {
    private func request(images: String?, extra: String = "") throws -> SystemOneRequest {
        let field = images.map { #","images":\#($0)"# } ?? ""
        let text =
            #"{"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"Red?"}}\#(field)\#(extra)}"#
        return try .parse(Data(text.utf8))
    }

    private func refusal(_ images: String, contains expected: String) {
        do {
            _ = try request(images: images)
            Issue.record("Accepted images: \(images.prefix(80))")
        }
        catch let DecisionWireError.invalid(message) {
            #expect(message.contains(expected), "\(message)")
        }
        catch {
            Issue.record("Unexpected error \(error)")
        }
    }

    @Test func bothItemFormsGiveTheSameImageAndDataPrefixIsCaseInsensitive() throws {
        let png = try image(.png, width: 8, height: 8)
        let b64 = png.base64EncodedString()
        let url = try request(images: #"["data:image/png;base64,\#(b64)"]"#).decision
        let upper = try request(images: #"["DATA:image/PNG;base64,\#(b64)"]"#).decision
        let object = try request(images: #"[{"content_type":"image/png","base64":"\#(b64)"}]"#).decision
        #expect(url.images == [DecisionImage(data: png, mediaType: "image/png")])
        #expect(url == object)
        #expect(url == upper)
    }

    @Test func emptyOrNullImagesEqualNoField() throws {
        let none = try request(images: nil).decision
        #expect(try request(images: "[]").decision == none)
        #expect(try request(images: "null").decision == none)
        #expect(none.images.isEmpty)
    }

    @Test func refusesEveryOutOfContractInput() throws {
        let png = try image(.png, width: 8, height: 8).base64EncodedString()
        let jpeg = try image(.jpeg, width: 8, height: 8).base64EncodedString()
        let gif = try image(.gif, width: 8, height: 8).base64EncodedString()
        let five = Array(repeating: #""data:image/png;base64,\#(png)""#, count: 5).joined(separator: ",")
        refusal("[\(five)]", contains: "at most 4 images")
        refusal(#"["https://example.com/a.png"]"#, contains: "remote URLs are not accepted")
        refusal(#"["data:image/gif;base64,\#(gif)"]"#, contains: "data URL must be image/png")
        // Declared types outside the allow-list are refused by the declaration, even when the bytes are a PNG.
        refusal(#"["data:image/gif;base64,\#(png)"]"#, contains: "data URL must be image/png")
        refusal(#"[{"content_type":"image/gif","base64":"\#(png)"}]"#, contains: "content_type must be image/png")
        refusal(#"[{"content_type":"image/png","base64":"\#(gif)"}]"#, contains: "PNG, JPEG, or WebP data")
        refusal(
            #"[{"content_type":"image/png","base64":"\#(jpeg)"}]"#,
            contains: "declared image/png but contains image/jpeg"
        )
        refusal(#"["data:image/png;base64,***not base64***"]"#, contains: "not valid base64")
        refusal(#"["data:image/png,\#(png)"]"#, contains: "base64 data URL")
        refusal(#"[{"content_type":"image/png"}]"#, contains: "content_type and base64")
        refusal(
            #"[{"content_type":"image/png","base64":"\#(png)","url":"x"}]"#,
            contains: "Unknown images[0] field 'url'"
        )
        refusal(#"[42]"#, contains: "data URL string or an object")
        refusal(#"{"a":1}"#, contains: "images must be an array")
        // A JPEG signature followed by garbage: right magic, not a decodable image.
        let fake = (Data([0xFF, 0xD8, 0xFF, 0xE0]) + Data(repeating: 7, count: 64)).base64EncodedString()
        refusal(#"["data:image/jpeg;base64,\#(fake)"]"#, contains: "not a decodable image")
    }

    @Test func sizeLimitsUseDecodedBytesAndHeaderPixels() throws {
        // Over 16 megapixels in a small file: refused from the header, without decoding pixels.
        let bomb = try image(.png, width: 5000, height: 4000)
        #expect(bomb.count < 4 * 1024 * 1024)
        refusal(#"["data:image/png;base64,\#(bomb.base64EncodedString())"]"#, contains: "larger than 16 megapixels")
        // Over 4 MiB decoded in one image.
        let big = try image(.png, width: 3000, height: 3000, noise: true)
        #expect(big.count > 4 * 1024 * 1024)
        refusal(#"["data:image/png;base64,\#(big.base64EncodedString())"]"#, contains: "larger than 4 MiB")
        // Over 8 MiB in total from three images under 4 MiB each.
        let part = try image(.png, width: 1050, height: 1050, noise: true)
        #expect(part.count < 4 * 1024 * 1024 && part.count * 3 > 8 * 1024 * 1024)
        let three = Array(repeating: #""data:image/png;base64,\#(part.base64EncodedString())""#, count: 3)
            .joined(separator: ",")
        refusal("[\(three)]", contains: "exceed 8 MiB in total")
    }

    @Test func truncatedImagesAreRefused() throws {
        // ImageIO accepts the header of a cut PNG and decodes the missing rows as black.
        let full = try image(.png, width: 512, height: 512, noise: true)
        _ = try request(images: #"["data:image/png;base64,\#(full.base64EncodedString())"]"#)
        for cut in [full.count / 2, full.count * 6 / 10, full.count - 13, full.count - 12, full.count - 1] {
            let part = full.prefix(cut).base64EncodedString()
            refusal(#"["data:image/png;base64,\#(part)"]"#, contains: "truncated or incomplete")
        }
        let webp = Data(base64Encoded: "UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==")!
        let cutWebp = webp.prefix(webp.count - 2).base64EncodedString()
        refusal(#"["data:image/webp;base64,\#(cutWebp)"]"#, contains: "truncated or incomplete")
    }

    @Test func handlerRefusesBodiesOver13MiB() async throws {
        #expect(try await handlerStatus(bodyBytes: SystemOneRequest.maximumBodyBytes + 1) == .payloadTooLarge)
        // Exactly the limit passes the size check and fails later as invalid JSON.
        #expect(try await handlerStatus(bodyBytes: SystemOneRequest.maximumBodyBytes) == .badRequest)
    }

    private func handlerStatus(bodyBytes: Int) async throws -> HTTPResponseStatus? {
        let channel = NIOAsyncTestingChannel()
        var body = channel.allocator.buffer(capacity: bodyBytes)
        body.writeRepeatingByte(UInt8(ascii: " "), count: bodyBytes)
        let head = HTTPRequestHead(version: .http1_1, method: .POST, uri: "/v1/systemone")
        await SystemOneHandler.handle(requestHead: head, body: body, channel: channel)
        guard case let .head(response)? = try await channel.readOutbound(as: HTTPServerResponsePart.self) else {
            return nil
        }

        return response.status
    }

    @Test func webpIsAcceptedBySignature() throws {
        // A 1x1 lossless WebP.
        let webp = "UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA=="
        let decision = try request(images: #"["data:image/webp;base64,\#(webp)"]"#).decision
        #expect(decision.images.first?.mediaType == "image/webp")
        #expect(SystemOneImages.sniffedMediaType(Data(base64Encoded: webp)!) == "image/webp")
    }

    @Test func videoIsRefusedNotIgnored() {
        do {
            _ = try request(images: nil, extra: #","video":"data:video/mp4;base64,AAAA""#)
            Issue.record("video was accepted")
        }
        catch let DecisionWireError.invalid(message) {
            #expect(message.contains("video is not supported"))
        }
        catch {
            Issue.record("Unexpected error \(error)")
        }
    }

    @Test func decisionsRouteStillRefusesImages() throws {
        let body: [String: JSONValue] = [
            "model": .string("m"), "input": .string("x"),
            "questions": .array([.object(["id": .string("q"), "type": .string("yes_no"), "question": .string("?")])]),
            "images": .array([.string("data:image/png;base64,AAAA")])
        ]
        #expect(throws: DecisionWireError.self) {
            _ = try DecisionsHandler.parse(body)
        }
    }

    /// A PNG, JPEG or GIF of the given size; `noise` makes it incompressible.
    private func image(_ type: UTType, width: Int, height: Int, noise: Bool = false) throws -> Data {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )
        else {
            throw DecisionWireError.invalid("context")
        }

        if noise, let data = context.data {
            let count = context.bytesPerRow * height
            let buffer = data.bindMemory(to: UInt8.self, capacity: count)
            var state: UInt64 = 0x9E37_79B9_7F4A_7C15
            for i in 0 ..< count {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                buffer[i] = UInt8(truncatingIfNeeded: state >> 33)
            }
        }
        else {
            context.setFillColor(CGColor(red: 0.9, green: 0.1, blue: 0.1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        }
        guard let cgImage = context.makeImage() else {
            throw DecisionWireError.invalid("image")
        }

        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, type.identifier as CFString, 1, nil) else {
            throw DecisionWireError.invalid("destination")
        }

        CGImageDestinationAddImage(destination, cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw DecisionWireError.invalid("finalize")
        }

        return output as Data
    }
}
