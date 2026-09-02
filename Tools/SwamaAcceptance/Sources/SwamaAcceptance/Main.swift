import Darwin
import SwamaAcceptanceKit

@main
struct SwamaAcceptanceMain {
    static func main() async {
        await exit(AcceptanceCLI.run(Array(CommandLine.arguments.dropFirst())))
    }
}
