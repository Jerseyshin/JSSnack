//
//  ViewController.swift
//  JSSnackDetection
//
//  Created by 辛泽西 on 2021/1/25.
//


import UIKit
import CoreMedia
import CoreML
import Vision

class ViewController: UIViewController {

    @IBOutlet weak var videoView: UIView!
    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var confidenceLabel: UILabel!
    
    let labels = [
        "apple","banana","cake","candy","carrot","cookie","doughnut","grape","hot dog",
        "ice cream","juice","muffin","orange","pineapple","popcorn","pretzel","salad",
        "strawberry","waffle","watermelon",
    ]
    
    // for video capturing
    var videoCapturer: VideoCapture!
    let semphore = DispatchSemaphore(value: ViewController.maxInflightBuffer)
    var inflightBuffer = 0
    static let maxInflightBuffer = 2
    
    lazy var classificationRequest: VNCoreMLRequest = {
        do{
            let classifier = try snack_localization(configuration: MLModelConfiguration())
            
            let model = try VNCoreMLModel(for: classifier.model)
            let request = VNCoreMLRequest(model: model, completionHandler: {
                [weak self] request,error in
                self?.processObservations(for: request, error: error)
            })
            request.imageCropAndScaleOption = .centerCrop
            return request
            
        } catch {
            fatalError("Failed to create request")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        self.setUpCamera()
    }
    
    func setUpCamera() {
        self.videoCapturer = VideoCapture()
        self.videoCapturer.delegate = self
        
        videoCapturer.frameInterval = 1
        videoCapturer.setUp(sessionPreset: .high, completion: {
            success in
            if success {
                if self.videoCapturer.previewLayer != nil {
                    self.view.layer.addSublayer(self.videoCapturer.previewLayer)
                    self.videoCapturer.previewLayer.frame = self.view.layer.frame
                    self.videoCapturer.previewLayer.addSublayer(self.resultLabel.layer)
                    self.videoCapturer.previewLayer.addSublayer(self.confidenceLabel.layer)
                    self.videoCapturer.start()
                }
            }
            else {
                print("Video capturer set up failed")
            }
        })
    }
    
    


}

extension ViewController: VideoCaptureDelegate {
    func videoCapture(capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
        self.classify(sampleBuffer: sampleBuffer)
    }
}


extension ViewController {
    func classify(sampleBuffer: CMSampleBuffer) {
        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            semphore.wait()
            inflightBuffer += 1
            if inflightBuffer >= ViewController.maxInflightBuffer {
                inflightBuffer = 0
            }
            DispatchQueue.main.async {
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
                do {
                    try handler.perform([self.classificationRequest])
                } catch {
                    print("Failed to perform classification: \(error)")
                }
                self.semphore.signal()
            }
            
        } else {
            print("Create pixel buffer failed")
        }
    }
}

extension ViewController {
    func DisView(flag: Bool){
        self.videoView.layer.borderWidth = 8
        if(flag){
            videoView.layer.borderColor = #colorLiteral(red:1, green:1, blue:1, alpha:1)
        }else{
            videoView.layer.borderColor = #colorLiteral(red:0, green:0, blue:0, alpha:0)
        }
    }
    
    func DisResult(flag: Bool){
        self.resultLabel.layer.borderWidth = 8
        if(flag){
            resultLabel.layer.borderColor = #colorLiteral(red:1, green:1, blue:1, alpha:1)
        }else{
            resultLabel.layer.borderColor = #colorLiteral(red:0, green:0, blue:0, alpha:0)
        }
    }
    
    func DisConfidence(flag: Bool){
        self.confidenceLabel.layer.borderWidth = 8
        if(flag){
            confidenceLabel.layer.borderColor = #colorLiteral(red:1, green:1, blue:1, alpha:1)
        }else{
            confidenceLabel.layer.borderColor = #colorLiteral(red:0, green:0, blue:0, alpha:0)
        }
    }
    
    func getRect(rect_values: MLMultiArray) -> (CGRect, CGRect, CGRect){
        let width = self.videoCapturer.previewLayer.frame.width
        let height = self.videoCapturer.previewLayer.frame.height
        
        let x_min = CGFloat(truncating: rect_values[0]) * width
        let x_max = CGFloat(truncating: rect_values[1]) * width
        let y_min = CGFloat(truncating: rect_values[2]) * height
        let y_max = CGFloat(truncating: rect_values[3]) * height
        
        let view_rect = CGRect(x: x_min, y: y_min, width: x_max - x_min, height: y_max - y_min)
        let res_rect = CGRect(x: x_min, y: y_min + CGFloat(-20), width: CGFloat(200), height: CGFloat(20))
        let con_rect = CGRect(x: x_min, y: y_min + CGFloat(-40), width: CGFloat(200), height: CGFloat(20))
        return(view_rect, res_rect, con_rect)
    }
    
    func processObservations(for request: VNRequest, error: Error?) {
        if let results = request.results as? [VNCoreMLFeatureValueObservation] {
            if results.isEmpty {
                self.DisView(flag: false)
                self.DisResult(flag: false)
                self.DisConfidence(flag: false)
            } else {
                //let result = results[0].identifier
                //let confidence = results[0].confidence
                //self.resultLabel.text = result
                //self.confidenceLabel.text = String(format: "%.1f%%", confidence * 100)
                
                let proArray = results[0]
                //输出1:20个概率
                let posArray = results[1]
                //输出2:4个位置
                let pro_values = proArray.featureValue.multiArrayValue!
                let pos_values = posArray.featureValue.multiArrayValue!
                
                //获取概率最大的食物
                let n = self.labels.count
                var classes = [Float](repeating: 0, count: n)
                for c in 0..<n{
                    classes[c] = Float(truncating: pro_values[c])
                }
                let max_idx = classes.firstIndex(of: classes.max()!)!
                
                //获取位置
                var view_rect = CGRect()
                var res_rect = CGRect()
                var con_rect = CGRect()
                (view_rect, res_rect, con_rect) = getRect(rect_values: pos_values)
                
                
                self.DisView(flag: true)
                self.DisResult(flag: true)
                self.DisConfidence(flag: true)
                
                DispatchQueue.main.async {
                    self.videoView.layer.frame = view_rect
                    self.resultLabel.text = self.labels[max_idx]
                    self.resultLabel.layer.frame = res_rect
                    self.confidenceLabel.text = String(classes[max_idx])
                    self.confidenceLabel.layer.frame = con_rect
                }
            }
        } else if let error = error {
            self.resultLabel.text = "Error: \(error.localizedDescription)"
        } else {
            self.resultLabel.text = "???"
        }
    }
}
