//  MagDirectionApp.swift
//  Real‑time magnet position classification using calibrated device‑frame magnetic field (device_x, device_y, device_z)
//
//  ▶︎ 手順
//    1. ContentView1.swift で提供されている MagneticFieldProcessor を **そのまま** プロジェクトに含める。
//       （地磁気キャリブレーション機能付き。currentMagneticField が μT 単位の device‑frame 磁場を公開）
//    2. Create ML Tabular Classifier で学習したモデル（入力: device_x, device_y, device_z / 出力: TargetDirection）
//       を Xcode プロジェクトに追加し、ここでは **MagDirectionClassifier.mlmodel** とする。
//    3. 本ファイルを追加してビルド → 実機で較正 → 方向判定を確認。
//
//  ©2025 Kazuya Ishikawa – MIT License

import SwiftUI
import CoreML
import Combine
import simd

// MARK: - 方向定義
enum Direction: String, CaseIterable, Identifiable {
    case top = "Top"
    case right = "Right"
    case origin = "Origin"
    case left = "Left"
    case down = "Down"

    var id: String { rawValue }

    var color: Color {
        switch self {
        case .top:    return .blue
        case .right:  return .green
        case .origin: return .gray
        case .left:   return .orange
        case .down:   return .red
        }
    }

    var systemImage: String {
        switch self {
        case .top:    return "arrow.up.circle.fill"
        case .right:  return "arrow.right.circle.fill"
        case .origin: return "circle.fill"
        case .left:   return "arrow.left.circle.fill"
        case .down:   return "arrow.down.circle.fill"
        }
    }

    /// UI レイアウト用オフセット割合 (‑1〜+1)
    var indicatorOffset: CGPoint {
        switch self {
        case .top:    return .init(x:  0, y:  1)
        case .right:  return .init(x:  1, y:  0)
        case .origin: return .init(x:  0, y:  0)
        case .left:   return .init(x: ‑1, y:  0)
        case .down:   return .init(x:  0, y: ‑1)
        }
    }
}

// MARK: - 方向予測器（device_x,y,z → TargetDirection）
final class DirectionPredictor: ObservableObject {
    @Published var direction: Direction = .origin
    @Published var confidence: Double = 0
    @Published var probabilities: [Direction: Double] = [:]
    @Published var status: String = "Loading model…"

    private var model: MagDirectionClassifier? = nil
    private var cancellables = Set<AnyCancellable>()

    init(processor: MagneticFieldProcessor) {
        loadModel()
        bind(to: processor)
    }

    // モデルロード
    private func loadModel() {
        do {
            model = try MagDirectionClassifier(configuration: .init())
            status = "Model ready ✓"
        } catch {
            status = "Model load failed: \(error.localizedDescription)"
            print("[DirectionPredictor] \(status)")
        }
    }

    // MagneticFieldProcessor から値を購読
    private func bind(to processor: MagneticFieldProcessor) {
        processor.$currentMagneticField
            .receive(on: DispatchQueue.global(qos: .userInitiated))
            .sink { [weak self] v in
                self?.predict(x: v.x, y: v.y, z: v.z)
            }
            .store(in: &cancellables)
    }

    // 予測実行
    private func predict(x: Double, y: Double, z: Double) {
        guard let model else { return }
        do {
            let output = try model.prediction(device_x: x, device_y: y, device_z: z)
            let dir = Direction(rawValue: output.TargetDirection) ?? .origin
            var probs: [Direction: Double] = [:]
            for d in Direction.allCases {
                probs[d] = output.TargetDirectionProbability[d.rawValue] ?? 0
            }
            DispatchQueue.main.async {
                self.direction   = dir
                self.probabilities = probs
                self.confidence  = probs[dir] ?? 0
            }
        } catch {
            DispatchQueue.main.async { self.status = "Prediction error: \(error.localizedDescription)" }
        }
    }
}

// MARK: - 方向表示ビュー
struct DirectionDisplay: View {
    let direction: Direction
    let confidence: Double
    let size: CGFloat = 260

    var body: some View {
        ZStack {
            Circle().stroke(Color.gray, lineWidth: 3).frame(width: size, height: size)
            ForEach(Direction.allCases) { dir in
                let offset = CGSize(width: dir.indicatorOffset.x * (size / 2 ‑ 30),
                                     height: ‑dir.indicatorOffset.y * (size / 2 ‑ 30))
                Circle()
                    .fill(dir == direction ? dir.color : Color.gray.opacity(0.3))
                    .frame(width: dir == direction ? 42 : 30, height: dir == direction ? 42 : 30)
                    .overlay(Image(systemName: dir.systemImage).font(.system(size: dir == direction ? 22 : 16)).foregroundColor(.white))
                    .offset(offset)
            }
            VStack(spacing: 8) {
                Image(systemName: direction.systemImage).font(.system(size: 60)).foregroundColor(direction.color)
                Text(direction.rawValue).font(.title2).bold().foregroundColor(direction.color)
                Text(String(format: "%.1f%%", confidence * 100)).font(.caption).foregroundColor(.secondary)
            }
        }
    }
}

// MARK: - メインビュー（UI＆操作）
struct ContentView: View {
    @StateObject private var processor = MagneticFieldProcessor()
    @StateObject private var predictor: DirectionPredictor

    init() {
        // DirectionPredictor は MagneticFieldProcessor を渡して初期化
        let p = MagneticFieldProcessor()
        _processor = StateObject(wrappedValue: p)
        _predictor = StateObject(wrappedValue: DirectionPredictor(processor: p))
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 25) {
                    Text("Magnet Direction Classifier")
                        .font(.largeTitle).bold()

                    // 方向表示
                    DirectionDisplay(direction: predictor.direction, confidence: predictor.confidence)
                        .padding(.top)

                    // ステータス
                    GroupBox("System Status") {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(predictor.status)
                            Text(processor.debugInfo).font(.caption)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    // キャリブレーション＆記録系ボタン（processor 既存実装を再利用）
                    VStack(spacing: 12) {
                        Button {
                            processor.startCalibration()
                        } label: {
                            Label("較正", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
                        }
                        .disabled(processor.isCalibrating || processor.isRecording)
                        .buttonStyle(.borderedProminent)

                        Button {
                            processor.toggleRecording()
                        } label: {
                            Label(processor.isRecording ? "記録停止" : "記録開始",
                                  systemImage: processor.isRecording ? "stop.circle.fill" : "record.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .tint(processor.isRecording ? .red : .green)
                        .disabled(processor.isCalibrating)
                        .buttonStyle(.bordered)
                    }

                    Spacer(minLength: 50)
                }
                .padding()
            }
            .navigationBarHidden(true)
        }
    }
}

// MARK: - プレビュー
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
