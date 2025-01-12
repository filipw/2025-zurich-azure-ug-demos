import Foundation

let sourceFileDir = (#file as NSString).deletingLastPathComponent
let basePath = ((sourceFileDir as NSString).appendingPathComponent("../../../../../fine-tuning/fused_model") as NSString).standardizingPath
let modelProvider = PhiModelProvider.fileSystem(
    indexPath: "\(basePath)/model.safetensors.index.json",
    configPath: "\(basePath)/config.json"
)

let inferenceOptionsBuilder = InferenceOptionsBuilder()
try! inferenceOptionsBuilder.withTemperature(temperature: 0.0)
try inferenceOptionsBuilder.withTokenCount(contextWindow: 50)
let inferenceOptions = try! inferenceOptionsBuilder.build()

let cacheDir = FileManager.default.currentDirectoryPath.appending("/.cache")

class ModelEventsHandler: PhiEventHandler {
    func onInferenceStarted() {}

    func onInferenceEnded() {}

    func onInferenceToken(token: String) {}

    func onModelLoaded() {
        print("""
 🧠 Model loaded!
****************************************
""")
    }
}

let modelBuilder = PhiEngineBuilder()
try! modelBuilder.withEventHandler(eventHandler: BoxedPhiEventHandler(handler: ModelEventsHandler()))
let gpuEnabled = try! modelBuilder.tryUseGpu()
try! modelBuilder.withModelProvider(modelProvider: modelProvider)

let model = try! modelBuilder.build(cacheDir: cacheDir)
let context = ConversationContext(messages: [], systemInstruction: "")

let prompts = [
    "Play alt rock",
    "too loud",
    "Skip this",
    "Next one please",
    "Change song",
    "play Comfortably Numb",
    "Go to last song",
    "What's the time?",
    "pause it",
    "make it quieter",
    "What should I eat?",
    "off",
    "Start focus music",
    "unmute",
    "What's your favorite color?",
]

for prompt in prompts {
    let result = try! model.runInference(promptText: prompt, conversationContext: context, inferenceOptions: inferenceOptions)
    print("\(prompt)   ->   \(result.resultText)")
}
