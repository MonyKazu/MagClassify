//
//  ContentView.swift
//  Real‑time magnet position classification using calibrated device‑frame magnetic field (device_x, device_y, device_z)
//
//  ▶︎ 手順
//    1. 本ファイルをプロジェクトに追加する。
//    2. Create ML で学習したモデル（入力: device_x, device_y, device_z / 出力: TargetDirection）
//       を Xcode プロジェクトに追加し、モデルのクラス名が参照と一致するようにする。
//       （例: MagDirectionClassifier.mlmodel）
//    3. ビルドし、実機で「較正」ボタンを押してキャリブレーションを実行する。
//    4. 磁石をiPhoneの背面に近づけ、方向が正しく判定されることを確認する。
//
//  ©2025 Kazuya Ishikawa – MIT License
//

import SwiftUI
import CoreML
import CoreMotion
import Combine
import simd

// MARK: - 磁気較正パラメータ
struct CalibrationParameters {
    var hardIronOffset: SIMD3<Double> = .zero
    var softIronMatrix: simd_double3x3 = matrix_identity_double3x3
    var isCalibrated: Bool = false
}

// MARK: - 磁気データ処理クラス
class MagneticFieldProcessor: ObservableObject {
    private let motionManager = CMMotionManager()
    private var calibrationData: [SIMD3<Double>] = []
    private var calibrationParameters = CalibrationParameters()

    // --- UI更新用プロパティ ---
    @Published var isCalibrating = false
    @Published var isRecording = false
    @Published var debugInfo = "較正ボタンを押してキャリブレーションを開始してください。"
    @Published var currentMagneticField = SIMD3<Double>.zero

    init() {
        setupMotionUpdates()
    }

    private func setupMotionUpdates() {
        guard motionManager.isDeviceMotionAvailable else {
            debugInfo = "Device Motion is not available."
            return
        }

        motionManager.deviceMotionUpdateInterval = 1.0 / 100.0 // 100Hz
        motionManager.showsDeviceMovementDisplay = true

        motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: .main) { [weak self] motion, error in
            guard let self = self, let motion = motion, error == nil else {
                print(error?.localizedDescription ?? "Unknown error")
                return
            }
            self.processMotionData(motion)
        }
    }

    private func processMotionData(_ motion: CMDeviceMotion) {
        let rawField = SIMD3<Double>(motion.magneticField.field.x, motion.magneticField.field.y, motion.magneticField.field.z)

        if isCalibrating {
            collectCalibrationData(rawField)
            let progress = min(1.0, Double(calibrationData.count) / (30.0 * 100.0))
            debugInfo = String(format: "8の字を描いて較正中... (%.0f%%)", progress * 100)
        } else if calibrationParameters.isCalibrated {
            let deviceFrameField = processField(rawField: rawField, attitude: motion.attitude)
            
            DispatchQueue.main.async {
                self.currentMagneticField = deviceFrameField
                let magnitude = simd_length(deviceFrameField)
                self.debugInfo = String(format: "較正完了 - 磁場強度: %.1f μT", magnitude)
            }
        }
    }
    
    // MARK: - 較正処理
    func startCalibration() {
        isCalibrating = true
        calibrationData.removeAll()
        calibrationParameters.isCalibrated = false
        debugInfo = "8の字を描くようにデバイスを動かしてください（30秒間）"

        DispatchQueue.main.asyncAfter(deadline: .now() + 30) { [weak self] in
            self?.completeCalibration()
        }
    }

    private func collectCalibrationData(_ field: SIMD3<Double>) {
        calibrationData.append(field)
    }

    private func completeCalibration() {
        isCalibrating = false
        guard calibrationData.count > 100 else {
            debugInfo = "較正失敗：データ不足。再度お試しください。"
            return
        }

        let params = calculateCalibrationParameters(from: calibrationData)
        self.calibrationParameters = params
        debugInfo = "較正完了。磁石を近づけてください。"
    }
    
    private func calculateCalibrationParameters(from data: [SIMD3<Double>]) -> CalibrationParameters {
        var params = CalibrationParameters()
        guard !data.isEmpty else { return params }

        let minX = data.map { $0.x }.min() ?? 0
        let maxX = data.map { $0.x }.max() ?? 0
        let minY = data.map { $0.y }.min() ?? 0
        let maxY = data.map { $0.y }.max() ?? 0
        let minZ = data.map { $0.z }.min() ?? 0
        let maxZ = data.map { $0.z }.max() ?? 0

        params.hardIronOffset = SIMD3<Double>((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2)

        let rangeX = max(maxX - minX, 1.0)
        let rangeY = max(maxY - minY, 1.0)
        let rangeZ = max(maxZ - minZ, 1.0)
        let avgRange = (rangeX + rangeY + rangeZ) / 3

        params.softIronMatrix = simd_double3x3(
            SIMD3<Double>(avgRange / rangeX, 0, 0),
            SIMD3<Double>(0, avgRange / rangeY, 0),
            SIMD3<Double>(0, 0, avgRange / rangeZ)
        )

        params.isCalibrated = true
        return params
    }

    // MARK: - 磁場処理（地磁気除去）
    private func processField(rawField: SIMD3<Double>, attitude: CMAttitude) -> SIMD3<Double> {
        let calibratedField = applyCalibration(rawField, with: calibrationParameters)
        let earthFieldInDeviceFrame = rotateToDeviceFrame(calibrationParameters.hardIronOffset, attitude: attitude)
        let magnetField = calibratedField - earthFieldInDeviceFrame
        return magnetField
    }
    
    private func applyCalibration(_ field: SIMD3<Double>, with params: CalibrationParameters) -> SIMD3<Double> {
        let corrected = field - params.hardIronOffset
        return params.softIronMatrix * corrected
    }
    
    private func rotateToDeviceFrame(_ vector: SIMD3<Double>, attitude: CMAttitude) -> SIMD3<Double> {
        let q = attitude.quaternion
        let rotation = simd_quatd(ix: q.x, iy: q.y, iz: q.z, r: q.w).inverse
        return rotation.act(vector)
    }

    func toggleRecording() {
        isRecording.toggle()
    }
}

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
    var indicatorOffset: CGPoint {
        switch self {
        case .top:    return .init(x:  0, y:  1)
        case .right:  return .init(x:  1, y:  0)
        case .origin: return .init(x:  0, y:  0)
        case .left:   return .init(x: -1, y:  0)
        case .down:   return .init(x:  0, y: -1)
        }
    }
}

// MARK: - 方向予測器（device_x,y,z → TargetDirection）
final class DirectionPredictor: ObservableObject {
    @Published var direction: Direction = .origin
    @Published var confidence: Double = 0
    @Published var probabilities: [Direction: Double] = [:]
    @Published var status: String = "Loading model…"

    private var model: MagClassify_cali? = nil
    private var cancellables = Set<AnyCancellable>()

    init(processor: MagneticFieldProcessor) {
        loadModel()
        bind(to: processor)
    }

    private func loadModel() {
        do {
            model = try MagClassify_cali(configuration: .init())
            status = "Model ready ✓"
        } catch {
            status = "Model load failed: \(error.localizedDescription)"
            print("[DirectionPredictor] \(status)")
        }
    }

    private func bind(to processor: MagneticFieldProcessor) {
        processor.$currentMagneticField
            .receive(on: DispatchQueue.global(qos: .userInitiated))
            .sink { [weak self] v in
                self?.predict(x: v.x, y: v.y, z: v.z)
            }
            .store(in: &cancellables)
    }

    private func predict(x: Double, y: Double, z: Double) {
        guard let model else { return }
        do {
            let output = try model.prediction(device_x: x, device_y: y, device_z: z)
            
            // Core MLモデルの出力名に合わせて修正 (例: targetDirection)
            let dir = Direction(rawValue: output.targetDirection) ?? .origin
            
            var probs: [Direction: Double] = [:]
            for d in Direction.allCases {
                // Core MLモデルの出力名に合わせて修正 (例: targetDirectionProbability)
                probs[d] = output.targetDirectionProbability[d.rawValue] ?? 0
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
                let offset = CGSize(width: dir.indicatorOffset.x * (size / 2 - 30),
                                     height: -dir.indicatorOffset.y * (size / 2 - 30))
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
    @StateObject private var processor: MagneticFieldProcessor
    @StateObject private var predictor: DirectionPredictor

    init() {
        // MagneticFieldProcessorとDirectionPredictorを初期化し、連携させる
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

                    DirectionDisplay(direction: predictor.direction, confidence: predictor.confidence)
                        .padding(.top)

                    GroupBox("System Status") {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(predictor.status)
                            Text(processor.debugInfo).font(.caption)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    VStack(spacing: 12) {
                        Button {
                            processor.startCalibration()
                        } label: {
                            Label("較正", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
                        }
                        .disabled(processor.isCalibrating)
                        .buttonStyle(.borderedProminent)
                    }
                    Spacer(minLength: 50)
                }
                .padding()
            }
            .navigationBarHidden(true)
        }
    }
}

// MARK: - Quaternion Extension
extension simd_quatd {
    func act(_ vector: SIMD3<Double>) -> SIMD3<Double> {
        let qv = self.imag
        let uv = cross(qv, vector)
        let uuv = cross(qv, uv)
        return vector + 2.0 * (self.real * uv + uuv)
    }
}

// MARK: - プレビュー
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
