"use client"
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Camera, Upload, Scan, Zap, Shield, AlertTriangle, CheckCircle, Loader2, RefreshCw, Info } from 'lucide-react';

const TomatoDiseaseClassifier = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [showCamera, setShowCamera] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);

  // Particle animation effect
  useEffect(() => {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const particles = [];
    for (let i = 0; i < 50; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: Math.random() * 2 + 0.5
      });
    }
    
    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      particles.forEach(particle => {
        particle.x += particle.vx;
        particle.y += particle.vy;
        
        if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
        if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;
        
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(34, 197, 94, 0.1)';
        ctx.fill();
      });
      
      requestAnimationFrame(animate);
    }
    
    animate();
    
    return () => {
      particles.length = 0;
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' }
      });
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setShowCamera(true);
    } catch (err) {
      setError('Unable to access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    setShowCamera(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        setSelectedFile(file);
        setPreview(URL.createObjectURL(blob));
        stopCamera();
      });
    }
  };

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    } else {
      setError('Please select a valid image file');
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Call the FastAPI backend
      const response = await fetch('https://tomato-disease-classifier-kl3p.onrender.com/classify', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze image');
      }

      if (data.success && data.data) {
        setResult(data.data);
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message || 'Failed to analyze image. Please check your connection and try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const getStatusColor = (disease) => {
    if (disease === 'Healthy') return 'text-green-400';
    if (disease === 'Late Blight' || disease === 'Yellow Leaf Curl Virus' || disease === 'Mosaic Virus') {
      return 'text-red-500'; // Critical diseases
    }
    return 'text-yellow-400'; // Medium severity diseases
  };

  const getStatusIcon = (disease) => {
    if (disease === 'Healthy') return <CheckCircle className="w-6 h-6" />;
    return <AlertTriangle className="w-6 h-6" />;
  };

  const getSeverityBadge = (result) => {
    if (!result.severity) return null;
    
    const severityColors = {
      'None': 'bg-green-500/20 text-green-400 border-green-500/30',
      'Medium': 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      'High': 'bg-orange-500/20 text-orange-400 border-orange-500/30',
      'Critical': 'bg-red-500/20 text-red-400 border-red-500/30'
    };

    return (
      <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${severityColors[result.severity] || 'bg-gray-500/20 text-gray-400 border-gray-500/30'}`}>
        {result.severity} Risk
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Particle Background */}
      <canvas 
        id="particle-canvas" 
        className="absolute inset-0 pointer-events-none"
        style={{ zIndex: 0 }}
      />
      
      {/* Animated Grid Background */}
      <div className="absolute inset-0 bg-gradient-to-r from-green-500/5 to-blue-500/5">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(rgba(34, 197, 94, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34, 197, 94, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
          animation: 'grid-move 20s linear infinite'
        }} />
      </div>

      {/* Main Content */}
      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              <Scan className="w-16 h-16 text-green-400 animate-pulse" />
              <div className="absolute -top-2 -right-2 w-6 h-6 bg-blue-500 rounded-full animate-ping" />
            </div>
          </div>
          <h1 className="text-6xl font-bold bg-gradient-to-r from-green-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-4">
            TomatoScan AI
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Advanced neural network-powered tomato disease detection system
          </p>
          <div className="flex items-center justify-center mt-4 space-x-2 text-green-400">
            <Zap className="w-5 h-5" />
            <span className="text-sm">Real-time AI Analysis</span>
          </div>
        </div>

        {/* Main Interface */}
        <div className="max-w-4xl mx-auto">
          {!showCamera && !preview && (
            <div 
              className={`relative group transition-all duration-300 ${
                dragActive ? 'scale-105' : 'hover:scale-102'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="relative backdrop-blur-xl bg-white/5 rounded-3xl border border-white/10 p-8 hover:bg-white/10 transition-all duration-500">
                {/* Glow effect */}
                <div className="absolute inset-0 rounded-3xl bg-gradient-to-r from-green-500/20 to-blue-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl" />
                
                <div className="relative z-10 text-center">
                  <div className="mb-8">
                    <Upload className="w-20 h-20 text-green-400 mx-auto mb-6 animate-bounce" />
                    <h3 className="text-2xl font-semibold text-white mb-4">
                      Upload Tomato Plant Image
                    </h3>
                    <p className="text-gray-300 mb-8">
                      Drag and drop an image or click to select from your device
                    </p>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="group relative px-8 py-4 bg-gradient-to-r from-green-500 to-blue-500 rounded-xl text-white font-semibold hover:from-green-400 hover:to-blue-400 transition-all duration-300 transform hover:scale-105 hover:shadow-xl hover:shadow-green-500/25"
                    >
                      <Upload className="w-5 h-5 inline mr-2" />
                      Choose File
                      <div className="absolute inset-0 rounded-xl bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    </button>
                    
                    <button
                      onClick={startCamera}
                      className="group relative px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-semibold hover:from-purple-400 hover:to-pink-400 transition-all duration-300 transform hover:scale-105 hover:shadow-xl hover:shadow-purple-500/25"
                    >
                      <Camera className="w-5 h-5 inline mr-2" />
                      Use Camera
                      <div className="absolute inset-0 rounded-xl bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Camera View */}
          {showCamera && (
            <div className="backdrop-blur-xl bg-white/5 rounded-3xl border border-white/10 p-6 mb-8">
              <div className="text-center mb-4">
                <h3 className="text-2xl font-semibold text-white mb-2">Camera View</h3>
                <p className="text-gray-300">Position your tomato plant in the frame</p>
              </div>
              
              <div className="relative rounded-2xl overflow-hidden bg-black/20 mb-4">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="w-full h-80 object-cover rounded-2xl"
                />
                <div className="absolute inset-0 border-2 border-green-400/50 rounded-2xl pointer-events-none">
                  <div className="absolute top-4 left-4 w-8 h-8 border-l-2 border-t-2 border-green-400" />
                  <div className="absolute top-4 right-4 w-8 h-8 border-r-2 border-t-2 border-green-400" />
                  <div className="absolute bottom-4 left-4 w-8 h-8 border-l-2 border-b-2 border-green-400" />
                  <div className="absolute bottom-4 right-4 w-8 h-8 border-r-2 border-b-2 border-green-400" />
                </div>
              </div>
              
              <div className="flex justify-center gap-4">
                <button
                  onClick={capturePhoto}
                  className="px-6 py-3 bg-gradient-to-r from-green-500 to-blue-500 rounded-xl text-white font-semibold hover:from-green-400 hover:to-blue-400 transition-all duration-300 transform hover:scale-105"
                >
                  <Camera className="w-5 h-5 inline mr-2" />
                  Capture
                </button>
                <button
                  onClick={stopCamera}
                  className="px-6 py-3 bg-gray-600 hover:bg-gray-500 rounded-xl text-white font-semibold transition-all duration-300"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Image Preview & Analysis */}
          {preview && (
            <div className="backdrop-blur-xl bg-white/5 rounded-3xl border border-white/10 p-6 mb-8">
              <div className="grid lg:grid-cols-2 gap-8">
                {/* Image Preview */}
                <div className="space-y-4">
                  <h3 className="text-xl font-semibold text-white flex items-center">
                    <Shield className="w-5 h-5 mr-2 text-green-400" />
                    Image Analysis
                  </h3>
                  <div className="relative group">
                    <img
                      src={preview}
                      alt="Selected tomato"
                      className="w-full h-80 object-cover rounded-2xl border-2 border-white/10 group-hover:border-green-400/50 transition-all duration-300"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  </div>
                  
                  <div className="flex gap-3">
                    <button
                      onClick={analyzeImage}
                      disabled={isAnalyzing}
                      className="flex-1 px-6 py-3 bg-gradient-to-r from-green-500 to-blue-500 rounded-xl text-white font-semibold hover:from-green-400 hover:to-blue-400 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2 className="w-5 h-5 inline mr-2 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Scan className="w-5 h-5 inline mr-2" />
                          Analyze Disease
                        </>
                      )}
                    </button>
                    
                    <button
                      onClick={resetAnalysis}
                      className="px-6 py-3 bg-gray-600 hover:bg-gray-500 rounded-xl text-white font-semibold transition-all duration-300"
                    >
                      <RefreshCw className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                {/* Results */}
                <div className="space-y-4">
                  <h3 className="text-xl font-semibold text-white flex items-center">
                    <Info className="w-5 h-5 mr-2 text-blue-400" />
                    Diagnosis Results
                  </h3>
                  
                  {isAnalyzing && (
                    <div className="backdrop-blur-xl bg-white/5 rounded-2xl border border-white/10 p-6">
                      <div className="text-center">
                        <Loader2 className="w-12 h-12 text-green-400 mx-auto mb-4 animate-spin" />
                        <p className="text-white">AI is analyzing your image...</p>
                        <div className="w-full bg-gray-700 rounded-full h-2 mt-4">
                          <div className="bg-gradient-to-r from-green-400 to-blue-400 h-2 rounded-full animate-pulse" style={{width: '60%'}} />
                        </div>
                      </div>
                    </div>
                  )}

                  {result && (
                    <div className="backdrop-blur-xl bg-white/5 rounded-2xl border border-white/10 p-6 space-y-4">
                      <div className="flex items-center justify-between mb-4">
                        <div className={`flex items-center ${getStatusColor(result.disease)}`}>
                          {getStatusIcon(result.disease)}
                          <span className="ml-2 text-xl font-semibold">{result.disease}</span>
                        </div>
                        <div className="text-right flex flex-col items-end gap-2">
                          <div>
                            <div className="text-sm text-gray-400">Confidence</div>
                            <div className="text-xl font-bold text-white">{result.confidence}%</div>
                          </div>
                          {getSeverityBadge(result)}
                        </div>
                      </div>
                      
                      <div className="w-full bg-gray-700 rounded-full h-3">
                        <div 
                          className="bg-gradient-to-r from-green-400 to-blue-400 h-3 rounded-full transition-all duration-1000"
                          style={{width: `${result.confidence}%`}}
                        />
                      </div>
                      
                      <p className="text-gray-300">{result.description}</p>
                      
                      {result.recommendations && result.recommendations.length > 0 && (
                        <div>
                          <h4 className="text-white font-semibold mb-3 flex items-center">
                            <Shield className="w-4 h-4 mr-2 text-green-400" />
                            Treatment Recommendations:
                          </h4>
                          <ul className="space-y-2">
                            {result.recommendations.map((rec, index) => (
                              <li key={index} className="text-gray-300 text-sm flex items-start p-3 rounded-lg bg-white/5 border border-white/10">
                                <span className="text-green-400 mr-3 mt-1">
                                  <CheckCircle className="w-4 h-4" />
                                </span>
                                {rec}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {error && (
                    <div className="backdrop-blur-xl bg-red-500/10 border border-red-500/20 rounded-2xl p-4">
                      <div className="flex items-center text-red-400">
                        <AlertTriangle className="w-5 h-5 mr-2" />
                        <span>{error}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
        className="hidden"
      />

      {/* Hidden canvas for camera capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Custom styles */}
      <style jsx>{`
        @keyframes grid-move {
          0% { transform: translate(0, 0); }
          100% { transform: translate(50px, 50px); }
        }
        
        .hover\\:scale-102:hover {
          transform: scale(1.02);
        }
      `}</style>
    </div>
  );
};

export default TomatoDiseaseClassifier;
