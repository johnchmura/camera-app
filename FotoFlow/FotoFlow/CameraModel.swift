import AVFoundation
import UIKit
import Photos
import SwiftData
import CoreImage
import Vision
import SwiftUI

class CameraModel: NSObject, ObservableObject {
    private let session = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")

    @Published var isCameraReady = false
    @Published var capturedImage: UIImage?
    @Published var burstImages: [UIImage] = []

    // Zoom factor range is [1, 3]
    @Published var currentZoomFactor: CGFloat = 1.0
    private var currentDevice: AVCaptureDevice?

    // Vision detection
    private var faceDetectionRequest = VNDetectFaceRectanglesRequest()
    private var bodyDetectionRequest = VNDetectHumanRectanglesRequest()

    // Normalized face and body dimensions for the T shape
    @Published var maxFaceWidth: CGFloat = 0.0
    @Published var maxBodyHeight: CGFloat = 0.0

    // Overlays – we now drive "No subject found" from body detection.
    @Published var backupOverlay: Bool = false

    // T shape color (normal .green, or .blue if steady alignment)
    @Published var tColor: Color = .green
    
    // Controls whether the T overlay is enabled.
    @Published var tOverlayEnabled: Bool = true

    // Indicates whether the T alignment is steady (used to enable the Zoom & Burst button)
    @Published var isSteadyAlignment: Bool = false

    // Threshold-based logic – each tuple is (range, recommendedZoom)
    private let thresholds: [(range: Range<CGFloat>, recommendedZoom: CGFloat)] = [
        (0.0..<0.005, 3.0),
        (0.005..<0.01, 2.5),
        (0.01..<0.02, 2.0),
        (0.02..<0.03, 1.8),
        (0.03..<0.05, 1.5),
        (0.05..<0.07, 1.3),
        (0.07..<0.09, 1.1)
        // If area >= 0.09, we treat it as "too close" and show the backup overlay.
    ]
    private let tooCloseArea: CGFloat = 0.09

    // Face detection state
    private var lastFaceDetectionTime: Date?
    private let faceDetectionTimeout: TimeInterval = 1.0 // 1 second

    // "Steady alignment" logic
    private var steadyStartTime: Date?
    private let steadyDuration: TimeInterval = 2.0
    private var lastKnownBoundingBox: CGRect = .zero // stores the face or body bounding box

    // Track last body detection time for overlay display.
    private var lastBodyDetectionTime: Date?

    // Flag to track if we’re in the middle of the Zoom & Burst flow.
    private var isInZoomAndBurstFlow = false

    var modelContext: ModelContext?

    // Computed property: show "No subject found" if no body is detected for >0.5 seconds.
    var noSubjectOverlay: Bool {
        if let lastTime = lastBodyDetectionTime {
            return Date().timeIntervalSince(lastTime) > 1.0
        } else {
            return true
        }
    }

    override init() {
        super.init()
        checkPermissions()
        setupFaceDetectionRequest()
        setupBodyDetectionRequest()
    }

    // MARK: - Permissions
    func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            break
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { _ in }
        default:
            print("Camera access denied or restricted.")
        }
    }

    // MARK: - Session Setup
    func startSession() {
        sessionQueue.async {
            if !self.session.isRunning {
                if self.session.inputs.isEmpty {
                    self.setupSession()
                }
                self.session.startRunning()
                print("Camera session started")
            }
        }
    }

    func stopSession() {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
    }

    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .photo

        // Use wideAngle camera
        guard let wideAngle = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: wideAngle),
              session.canAddInput(input),
              session.canAddOutput(photoOutput) else {
            print("Failed to set up camera input/output.")
            session.commitConfiguration()
            return
        }

        session.addInput(input)
        session.addOutput(photoOutput)
        currentDevice = wideAngle

        if #available(iOS 16.0, *) {
            photoOutput.maxPhotoDimensions = CMVideoDimensions(width: 4032, height: 3024)
        } else {
            photoOutput.isHighResolutionCaptureEnabled = true
        }

        // Add video data output for Vision processing
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            videoDataOutput.videoSettings =
                [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoDataOutput.setSampleBufferDelegate(self, queue: sessionQueue)
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
        }

        session.commitConfiguration()

        DispatchQueue.main.async {
            self.isCameraReady = true
        }

        setZoom(factor: 1.0) // start at 1x
    }

    // MARK: - Vision Setup
    private func setupFaceDetectionRequest() {
        faceDetectionRequest = VNDetectFaceRectanglesRequest { [weak self] request, _ in
            guard let self = self else { return }
            if let faces = request.results as? [VNFaceObservation], !faces.isEmpty {
                // Pick the largest face.
                let largestFace = faces.max(by: { $0.boundingBox.width < $1.boundingBox.width })
                let maxWidth = largestFace?.boundingBox.width ?? 0.0
                if let largestFace = largestFace {
                    self.lastKnownBoundingBox = largestFace.boundingBox
                }
                DispatchQueue.main.async {
                    self.maxFaceWidth = maxWidth
                    self.lastFaceDetectionTime = Date()
                }
            }
        }
    }

    private func setupBodyDetectionRequest() {
        bodyDetectionRequest = VNDetectHumanRectanglesRequest { [weak self] request, _ in
            guard let self = self else { return }
            if let bodies = request.results as? [VNHumanObservation], !bodies.isEmpty {
                let largestBody = bodies.max(by: { $0.boundingBox.height < $1.boundingBox.height })
                let maxHeight = largestBody?.boundingBox.height ?? 0.0
                DispatchQueue.main.async {
                    self.maxBodyHeight = maxHeight
                    self.lastBodyDetectionTime = Date() // update detection time
                }
            }
        }
    }

    // Called for every video frame.
    private func processSampleBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        do {
            try requestHandler.perform([faceDetectionRequest, bodyDetectionRequest])
        } catch {
            print("Error performing vision detection: \(error)")
        }

        handleNoFaceIfTimeout()
        if faceIsDetected() {
            handleAreaBasedLogic()
            handleSteadyAlignmentCheck()
        }
    }

    // MARK: - Face Logic
    private func handleNoFaceIfTimeout() {
        if let lastTime = lastFaceDetectionTime {
            let elapsed = Date().timeIntervalSince(lastTime)
            if elapsed > faceDetectionTimeout {
                DispatchQueue.main.async {
                    self.maxFaceWidth = 0
                    self.backupOverlay = false
                }
            }
        } else {
            DispatchQueue.main.async {
                // No face ever detected; nothing to do.
            }
        }
    }

    private func faceIsDetected() -> Bool {
        return maxFaceWidth > 0
    }

    // MARK: - Area-based Logic
    private func handleAreaBasedLogic() {
        let area = maxFaceWidth * maxBodyHeight
        if area >= tooCloseArea {
            DispatchQueue.main.async {
                self.backupOverlay = true
            }
        } else {
            DispatchQueue.main.async {
                self.backupOverlay = false
            }
        }
    }

    // MARK: - Steady Alignment Check
    private func handleSteadyAlignmentCheck() {
        let centerX = lastKnownBoundingBox.midX
        let centerY = lastKnownBoundingBox.midY
        // Adjust target as needed (for example, for rule-of-thirds, use (0.5, 0.66)).
        let targetX: CGFloat = 0.5
        let targetY: CGFloat = 0.5
        let threshold: CGFloat = 0.05

        let dx = abs(centerX - targetX)
        let dy = abs(centerY - targetY)
        let aligned = (dx < threshold) && (dy < threshold)

        if aligned {
            if steadyStartTime == nil {
                steadyStartTime = Date()
                DispatchQueue.main.async {
                    self.isSteadyAlignment = false
                }
            } else {
                let duration = Date().timeIntervalSince(steadyStartTime!)
                if duration >= steadyDuration {
                    DispatchQueue.main.async {
                        self.tColor = .blue
                        self.isSteadyAlignment = true
                    }
                }
            }
        } else {
            steadyStartTime = nil
            DispatchQueue.main.async {
                self.tColor = .green
                self.isSteadyAlignment = false
            }
        }
    }

    // MARK: - Zoom & Burst Flow
    func beginZoomAndBurst() {
        // Only allow triggering if steady alignment is reached and not already in a burst flow.
        guard isSteadyAlignment, !isInZoomAndBurstFlow else {
            print("Zoom & Burst ignored; either not steady or already in burst flow.")
            return
        }
        // Immediately reset the steady flag to prevent multiple presses.
        DispatchQueue.main.async {
            self.isSteadyAlignment = false
        }
        isInZoomAndBurstFlow = true

        let area = maxFaceWidth * maxBodyHeight
        if let match = thresholds.first(where: { $0.range.contains(area) }) {
            let recommended = match.recommendedZoom
            if recommended > currentZoomFactor {
                setZoom(factor: recommended)
            }
        }
        // Trigger burst capture.
        captureBurstPhotos()

        // After a delay, reset zoom to default and clear the burst flow flag.
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            self.setZoom(factor: 1.0)
            self.tColor = .green
            self.isInZoomAndBurstFlow = false
            // Disable the T overlay for 0.5 seconds after reset.
            self.tOverlayEnabled = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                self.tOverlayEnabled = true
            }
        }
    }

    // MARK: - Zoom Methods
    func setZoom(factor: CGFloat) {
        let clamped = max(1.0, min(factor, 3.0))
        guard let device = currentDevice else {
            print("❌ No camera device available for zoom")
            return
        }
        do {
            try device.lockForConfiguration()
            let minZoom = max(device.minAvailableVideoZoomFactor, 1.0)
            let maxZoom = min(device.maxAvailableVideoZoomFactor, 3.0)
            device.videoZoomFactor = max(minZoom, min(clamped, maxZoom))
            device.unlockForConfiguration()

            DispatchQueue.main.async {
                self.currentZoomFactor = clamped
            }
            print("🔍 Zoom adjusted to \(clamped)x")
        } catch {
            print("❌ Failed to zoom: \(error.localizedDescription)")
        }
    }

    func incrementZoom() {
        var nextZoom = (currentZoomFactor * 10).rounded(.up) / 10 + 0.1
        if nextZoom < 1.0 { nextZoom = 1.0 }
        setZoom(factor: nextZoom)
    }

    func getSession() -> AVCaptureSession {
        return session
    }

    // MARK: - Photo Capture
    func capturePhoto() {
        let settings = AVCapturePhotoSettings()
        if #available(iOS 16.0, *) {
            // Additional settings if needed.
        } else {
            settings.isHighResolutionPhotoEnabled = true
        }
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    func captureBurstPhotos() {
        DispatchQueue.main.async {
            self.burstImages = []
        }
        let totalBurstCount = 5
        var burstCount = 0

        func captureNext() {
            guard burstCount < totalBurstCount else {
                print("✅ Finished burst of \(totalBurstCount) photos.")
                return
            }
            let settings = AVCapturePhotoSettings()
            photoOutput.capturePhoto(with: settings, delegate: self)
            burstCount += 1
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                captureNext()
            }
        }
        captureNext()
    }

    private func saveToPhotoLibrary(image: UIImage) {
        PHPhotoLibrary.requestAuthorization { status in
            if status == .authorized || status == .limited {
                PHPhotoLibrary.shared().performChanges({
                    PHAssetChangeRequest.creationRequestForAsset(from: image)
                }) { success, error in
                    if success {
                        print("✅ Photo saved to photo library")
                    } else if let error = error {
                        print("❌ Error saving photo: \(error.localizedDescription)")
                    }
                }
            } else {
                print("❌ Photo library access not granted")
            }
        }
    }

    private func saveToModel(date: Date) {
        guard let context = modelContext else {
            print("❌ modelContext is nil")
            return
        }
        let newEntry = CapturedPhoto(dateCaptured: date)
        context.insert(newEntry)
        print("📝 Saved date metadata to SwiftData")
    }
}

// MARK: - AVCapturePhotoCaptureDelegate
extension CameraModel: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        if let data = photo.fileDataRepresentation(),
           let image = UIImage(data: data) {
            DispatchQueue.main.async {
                self.capturedImage = image
                self.burstImages.append(image)
                let captureDate = Date()
                self.saveToPhotoLibrary(image: image)
                self.saveToModel(date: captureDate)
            }
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        processSampleBuffer(sampleBuffer)
    }
}
