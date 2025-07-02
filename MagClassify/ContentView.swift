import SwiftUI
import CoreMotion
import CoreML
import Combine
import simd
import UniformTypeIdentifiers

// MARK: - 磁気較正パラメータ
struct CalibrationParameters {
    var hardIronOffset: SIMD3<Double> = SIMD3(0, 0, 0)
    var softIronMatrix: simd_double3x3 = matrix_identity_double3x3
    var isCalibrated: Bool = false
}

// MARK: - 位置推定結果
enum MagnetPosition: String, CaseIterable {
    case top = "Top"
    case down = "Down"
    case left = "Left"
    case right = "Right"
    case center = "Center"
    case unknown = "Unknown"
    
    var displayName: String {
        switch self {
        case .top: return "上"
        case .down: return "下"
        case .left: return "左"
        case .right: return "右"
        case .center: return "中心"
        case .unknown: return "不明"
        }
    }
    
    var color: Color {
        switch self {
        case .top: return .blue
        case .down: return .orange
        case .left: return .green
        case .right: return .red
        case .center: return .gray
        case .unknown: return .secondary
        }
    }
}

// MARK: - 磁気データ処理クラス
class MagneticPositionClassifier: ObservableObject {
    private let motionManager = CMMotionManager()
    private var calibrationData: [SIMD3<Double>] = []
    private var calibrationParameters = CalibrationParameters()
    
    // MLモデル
    private var mlModel: MLModel?
    // ★ モデル名を定数として定義
    private let modelName = "MagClassify_cali"
    
    // --- UI更新用プロパティ ---
    @Published var isCalibrating = false
    @Published var debugInfo = "モデルを読み込み中..."
    @Published var currentPosition: MagnetPosition = .unknown
    @Published var confidence: Double = 0.0
    @Published var currentMagneticField = SIMD3<Double>(0, 0, 0)
    @Published var currentMagnitude: Double = 0.0
    @Published var isModelLoaded = false
    
    // 磁石検出用
    private let magnetDetectionThreshold: Double = 100.0 // μT
    @Published var magnetDetected = false
    
    init() {
        // ★ 初期化時にモデルを自動読み込み
        loadModel()
        setupMotionUpdates()
    }
    
    // MARK: - モデル読み込み (自動化)
    // ★ アプリバンドル内からモデルを読み込むように変更
    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            debugInfo = "モデルファイルが見つかりません: \(modelName).mlmodelc\nプロジェクトに追加されているか確認してください。"
            isModelLoaded = false
            return
        }
        
        do {
            mlModel = try MLModel(contentsOf: modelURL)
            isModelLoaded = true
            debugInfo = "モデル読み込み完了。較正ボタンを押してください。"
        } catch {
            debugInfo = "モデルの読み込みに失敗しました: \(error.localizedDescription)"
            isModelLoaded = false
        }
    }
    
    private func setupMotionUpdates() {
        guard motionManager.isDeviceMotionAvailable else {
            debugInfo = "Device Motion is not available."
            return
        }
        
        motionManager.deviceMotionUpdateInterval = 1.0 / 50.0 // 50Hz
        motionManager.showsDeviceMovementDisplay = true
        
        motionManager.startDeviceMotionUpdates(
            using: .xMagneticNorthZVertical,
            to: .main
        ) { [weak self] motion, error in
            guard let self = self,
                  let motion = motion,
                  error == nil else {
                print(error?.localizedDescription ?? "Unknown error")
                return
            }
            
            self.processMotionData(motion)
        }
    }
    
    private func processMotionData(_ motion: CMDeviceMotion) {
        let magneticField = motion.magneticField.field
        let rawField = SIMD3<Double>(magneticField.x, magneticField.y, magneticField.z)
        
        if isCalibrating {
            collectCalibrationData(rawField)
        } else if calibrationParameters.isCalibrated {
            if let processedData = processMagneticField(
                rawField: rawField,
                attitude: motion.attitude
            ) {
                // 磁石検出
                detectMagnet(magnitude: processedData.magnitude)
                
                // UIの更新
                DispatchQueue.main.async {
                    self.currentMagneticField = processedData.deviceFrame
                    self.currentMagnitude = processedData.magnitude
                }
                
                // 位置推定
                if magnetDetected && isModelLoaded {
                    classifyPosition(deviceX: processedData.deviceFrame.x,
                                     deviceY: processedData.deviceFrame.y,
                                     deviceZ: processedData.deviceFrame.z)
                } else if !magnetDetected {
                    DispatchQueue.main.async {
                        self.currentPosition = .unknown
                        self.confidence = 0.0
                    }
                }
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
            debugInfo = "較正失敗：データ不足"
            return
        }
        
        let parameters = calculateCalibrationParameters(from: calibrationData)
        calibrationParameters = parameters
        
        debugInfo = "較正完了。磁石を近づけて位置推定を開始してください。"
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
        
        params.hardIronOffset = SIMD3<Double>(
            (minX + maxX) / 2,
            (minY + maxY) / 2,
            (minZ + maxZ) / 2
        )
        
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
    
    // MARK: - 磁場処理
    private func processMagneticField(
        rawField: SIMD3<Double>,
        attitude: CMAttitude
    ) -> (deviceFrame: SIMD3<Double>, magnitude: Double)? {
        let calibratedField = applyCalibration(rawField, with: calibrationParameters)
        let worldFrameField = convertToWorldFrame(calibratedField, attitude: attitude)
        let deviceFrameField = convertToDeviceFrame(worldFrameField, attitude: attitude)
        let magnitude = simd_length(calibratedField)
        
        return (deviceFrame: deviceFrameField, magnitude: magnitude)
    }
    
    private func applyCalibration(_ field: SIMD3<Double>, with params: CalibrationParameters) -> SIMD3<Double> {
        let corrected = field - params.hardIronOffset
        return params.softIronMatrix * corrected
    }
    
    private func convertToWorldFrame(_ field: SIMD3<Double>, attitude: CMAttitude) -> SIMD3<Double> {
        let q = attitude.quaternion
        let rotation = simd_quatd(ix: q.x, iy: q.y, iz: q.z, r: q.w)
        return rotation.act(field)
    }
    
    private func convertToDeviceFrame(_ field: SIMD3<Double>, attitude: CMAttitude) -> SIMD3<Double> {
        let q = attitude.quaternion
        let rotation = simd_quatd(ix: q.x, iy: q.y, iz: q.z, r: q.w).inverse
        return rotation.act(field)
    }
    
    // MARK: - 磁石検出
    private func detectMagnet(magnitude: Double) {
        magnetDetected = magnitude > magnetDetectionThreshold
    }
    
    // MARK: - 位置推定
    private func classifyPosition(deviceX: Double, deviceY: Double, deviceZ: Double) {
        guard let model = mlModel else { return }
        
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "device_x": deviceX,
                "device_y": deviceY,
                "device_z": deviceZ
            ])
            
            let prediction = try model.prediction(from: input)
            
            if let targetDirection = prediction.featureValue(for: "targetDirection")?.stringValue {
                let position = MagnetPosition(rawValue: targetDirection) ?? .unknown
                
                var maxConfidence = 1.0
                if let probabilities = prediction.featureValue(for: "targetDirectionProbability")?.dictionaryValue {
                    maxConfidence = probabilities.values.compactMap { $0 as? Double }.max() ?? 1.0
                }
                
                DispatchQueue.main.async {
                    self.currentPosition = position
                    self.confidence = maxConfidence
                    self.debugInfo = "位置: \(position.displayName) (信頼度: \(Int(maxConfidence * 100))%)"
                }
            }
        } catch {
            DispatchQueue.main.async {
                self.debugInfo = "推論エラー: \(error.localizedDescription)"
            }
        }
    }
}


// MARK: - メインビュー
struct ContentView: View {
    @StateObject private var classifier = MagneticPositionClassifier()
    // ★ ファイルピッカー関連のプロパティを削除
    // @State private var showingFilePicker = false
    
    var body: some View {
        VStack(spacing: 20) {
            // タイトル
            Text("磁石位置推定システム")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            // モデル読み込み状態
            HStack {
                Image(systemName: classifier.isModelLoaded ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundColor(classifier.isModelLoaded ? .green : .red)
                Text(classifier.isModelLoaded ? "モデル読み込み済み" : "モデル未読み込み")
            }
            
            // 現在の位置表示
            if classifier.magnetDetected {
                VStack(spacing: 10) {
                    Text("検出位置")
                        .font(.headline)
                    
                    Text(classifier.currentPosition.displayName)
                        .font(.system(size: 48, weight: .bold))
                        .foregroundColor(classifier.currentPosition.color)
                    
                    // 信頼度表示
                    HStack {
                        Text("信頼度:")
                        ProgressView(value: classifier.confidence, total: 1.0)
                            .frame(width: 100)
                        Text("\(Int(classifier.confidence * 100))%")
                            .font(.system(.caption, design: .monospaced))
                    }
                }
                .padding()
                .background(classifier.currentPosition.color.opacity(0.1))
                .cornerRadius(15)
            } else {
                Text("磁石を近づけてください")
                    .font(.headline)
                    .foregroundColor(.secondary)
                    .padding()
            }
            
            // 磁場情報
            GroupBox("磁場情報") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("X:")
                        Text(String(format: "%6.1f μT", classifier.currentMagneticField.x))
                            .font(.system(.body, design: .monospaced))
                    }
                    HStack {
                        Text("Y:")
                        Text(String(format: "%6.1f μT", classifier.currentMagneticField.y))
                            .font(.system(.body, design: .monospaced))
                    }
                    HStack {
                        Text("Z:")
                        Text(String(format: "%6.1f μT", classifier.currentMagneticField.z))
                            .font(.system(.body, design: .monospaced))
                    }
                    Divider()
                    HStack {
                        Text("大きさ:")
                        Text(String(format: "%.1f μT", classifier.currentMagnitude))
                            .font(.system(.body, design: .monospaced))
                            .fontWeight(.bold)
                        Spacer()
                        if classifier.magnetDetected {
                            Label("磁石検出", systemImage: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.caption)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            
            // ステータス
            Text(classifier.debugInfo)
                .font(.caption)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            
            // コントロールボタン
            VStack(spacing: 15) {
                // ★ 「モデルを読み込む」ボタンを削除
                
                Button(action: { classifier.startCalibration() }) {
                    Label("較正", systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                }
                .disabled(classifier.isCalibrating || !classifier.isModelLoaded)
                .buttonStyle(.borderedProminent)
            }
            
            if classifier.isCalibrating {
                ProgressView("較正中...")
                    .padding()
            }
            
            Spacer()
        }
        .padding()
        // ★ .fileImporterモディファイアを削除
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

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
    
}
