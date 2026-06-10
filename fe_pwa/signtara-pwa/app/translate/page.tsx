"use client";

import { useState, useEffect, useRef } from "react";
import { ArrowLeft, Hand, Mic, Keyboard, Volume2, HelpCircle, Camera as CameraIcon, RefreshCw, Play, Square, X, Delete, SwitchCamera, History, Save, Trash2 } from "lucide-react";
import Link from "next/link";
import * as tf from '@tensorflow/tfjs';

// Tipe data untuk Riwayat Terjemahan
interface HistoryItem {
  id: string;
  text: string;
  type: "tuli" | "dengar";
  timestamp: number;
}

export default function TranslatePage() {
  // ==========================================
  // STATE & REF DECLARATIONS
  // ==========================================
  
  // State UI
  const [showTutorial, setShowTutorial] = useState(true);
  const [activeTab, setActiveTab] = useState("tuli");
  const [isTyping, setIsTyping] = useState(false);
  const [isCheckingMemory, setIsCheckingMemory] = useState(true);
  
  // State Hardware & Action Spotting
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("Menyiapkan...");
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user"); 

  // State Teman Dengar
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState(""); 
  
  // State AI & Terjemahan
  const [isModelReady, setIsModelReady] = useState(false);
  const [sentence, setSentence] = useState<string[]>([]); 
  const [isAutoSpeak, setIsAutoSpeak] = useState(false); // Memakai state bawaan yang bekerja
  const [textSize, setTextSize] = useState(40); 
  const videoRef = useRef<HTMLVideoElement>(null);

  // State Riwayat (History)
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  
  // Ref AI Core
  const modelRef = useRef<tf.LayersModel | null>(null);
  const classNamesRef = useRef<string[]>([]);
  const holisticRef = useRef<any>(null); 
  const isInitializingRef = useRef(false); // Gembok Anti-Crash WASM
  
  // Ref Logika State Machine
  const recentFeaturesRef = useRef<number[][]>([]); 
  const recentRawHandsRef = useRef<number[][]>([]); // Anti-Tremor
  const recordedSequenceRef = useRef<number[][]>([]);
  const isSigningRef = useRef(false);
  const isPredictingRef = useRef(false);
  const animationFrameId = useRef<number | null>(null);

  // Konfigurasi AI
  const MAX_FRAMES = 20;
  const MOTION_THRESHOLD = 0.015; 
  const CONFIDENCE_THRESHOLD = 0.85;

  // ==========================================
  // EFFECTS (LIFECYCLE)
  // ==========================================

  useEffect(() => {
    const hasSeenTutorial = localStorage.getItem("signtara_tutorial_seen");
    if (hasSeenTutorial === "true") setShowTutorial(false);
    
    // Load Riwayat
    const savedHistory = localStorage.getItem("signtara_history");
    if (savedHistory) setHistory(JSON.parse(savedHistory));

    setIsCheckingMemory(false);
  }, []);

  useEffect(() => {
    localStorage.setItem("signtara_history", JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    const loadAIModel = async () => {
      if (holisticRef.current || isInitializingRef.current) return;
      isInitializingRef.current = true;

      try {
        const classResponse = await fetch('/model/class_names.json');
        const classData = await classResponse.json();
        classNamesRef.current = Object.values(classData); 

        const model = await tf.loadLayersModel('/model/model.json');
        modelRef.current = model;

        const mediapipe = await import('@mediapipe/holistic');
        const Holistic = mediapipe.Holistic || (mediapipe as any).default.Holistic || (window as any).Holistic;

        const holistic = new Holistic({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        });
        
        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          refineFaceLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        holistic.onResults(handleHolisticResults);
        holisticRef.current = holistic;

        console.log("Sistem AI (TFJS & MediaPipe Holistic) Siap!");
        setIsModelReady(true);
        setCameraStatus("Siap");
      } catch (error) {
        console.error("Gagal memuat model AI:", error);
        isInitializingRef.current = false;
      }
    };

    loadAIModel();
    return () => { if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current); };
  }, []);

  useEffect(() => {
    if (activeTab === "dengar") stopCamera();
  }, [activeTab]);


  // ==========================================
  // FUNGSI PENDUKUNG KALIMAT & SUARA
  // ==========================================

  const handleUndo = () => setSentence(prev => prev.slice(0, -1));
  const handleClearSentence = () => setSentence([]);

  // Menggunakan fungsi text-to-speech yang sudah terbukti jalan di komputermu
  const speakText = (text: string) => {
    if ('speechSynthesis' in window && text.trim() !== "") {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'id-ID'; 
      window.speechSynthesis.speak(utterance);
    }
  };

  const speakFullSentence = () => {
    if (sentence.length > 0) speakText(sentence.join(" "));
  };

  const handleSaveHistory = (text: string, type: "tuli" | "dengar") => {
    if (!text.trim()) return;
    const newItem: HistoryItem = {
      id: Date.now().toString(),
      text: text.trim(),
      type,
      timestamp: Date.now(),
    };
    setHistory(prev => [newItem, ...prev]);
    alert("Berhasil disimpan ke Riwayat!");
  };


  // ==========================================
  // FUNGSI AI: PREPROCESSING & LOGIKA STATE
  // ==========================================

  const processSequence = (sequence: number[][], maxFrames: number) => {
    const n = sequence.length;
    if (n >= maxFrames) {
      const result = [];
      for (let i = 0; i < maxFrames; i++) {
        const idx = Math.floor((i * (n - 1)) / (maxFrames - 1));
        result.push(sequence[idx]);
      }
      return result;
    } else {
      const result = [...sequence];
      while (result.length < maxFrames) {
        result.push(sequence[sequence.length - 1]);
      }
      return result;
    }
  };

  const handleHolisticResults = async (results: any) => {
    const POSE_IDX = [0, 7, 8, 11, 12];
    const FACE_IDX = [10, 0, 17, 61, 291];
    
    let anchorX = 0, anchorY = 0, anchorZ = 0;
    if (results.poseLandmarks && results.poseLandmarks.length > POSE_IDX[0]) {
      const nose = results.poseLandmarks[POSE_IDX[0]];
      anchorX = nose.x; anchorY = nose.y; anchorZ = nose.z;
    }

    let feat: number[] = []; 
    let rawHands: number[] = []; 

    if (results.poseLandmarks) {
      POSE_IDX.forEach(i => {
        const lm = results.poseLandmarks[i] || {x: anchorX, y: anchorY, z: anchorZ};
        feat.push(lm.x - anchorX, lm.y - anchorY, lm.z - anchorZ);
      });
    } else { feat.push(...new Array(15).fill(0)); }

    if (results.faceLandmarks) {
      FACE_IDX.forEach(i => {
        const lm = results.faceLandmarks[i] || {x: anchorX, y: anchorY, z: anchorZ};
        feat.push(lm.x - anchorX, lm.y - anchorY, lm.z - anchorZ);
      });
    } else { feat.push(...new Array(15).fill(0)); }

    if (results.leftHandLandmarks) {
      results.leftHandLandmarks.forEach((lm: any) => {
        feat.push(lm.x - anchorX, lm.y - anchorY, lm.z - anchorZ); 
        rawHands.push(lm.x, lm.y, lm.z); 
      });
    } else { 
      feat.push(...new Array(63).fill(0)); 
      rawHands.push(...new Array(63).fill(0));
    }

    if (results.rightHandLandmarks) {
      results.rightHandLandmarks.forEach((lm: any) => {
        feat.push(lm.x - anchorX, lm.y - anchorY, lm.z - anchorZ); 
        rawHands.push(lm.x, lm.y, lm.z); 
      });
    } else { 
      feat.push(...new Array(63).fill(0)); 
      rawHands.push(...new Array(63).fill(0));
    }

    recentFeaturesRef.current.push(feat);
    if (recentFeaturesRef.current.length > 5) recentFeaturesRef.current.shift();

    recentRawHandsRef.current.push(rawHands);
    if (recentRawHandsRef.current.length > 5) recentRawHandsRef.current.shift();

    if (recentRawHandsRef.current.length === 5 && !isPredictingRef.current) {
      let totalVelocity = 0;
      
      for (let i = 1; i < recentRawHandsRef.current.length; i++) {
        let frameVelocity = 0;
        for (let j = 0; j < 126; j++) {
          let diff = recentRawHandsRef.current[i][j] - recentRawHandsRef.current[i-1][j];
          frameVelocity += diff * diff;
        }
        totalVelocity += Math.sqrt(frameVelocity);
      }
      let meanVelocity = totalVelocity / 4;

      let currentIsAutoSpeak = false;
      setIsAutoSpeak(prev => { currentIsAutoSpeak = prev; return prev; });

      if (meanVelocity > MOTION_THRESHOLD) {
        setCameraStatus("Merekam...");
        if (!isSigningRef.current) {
          isSigningRef.current = true;
          recordedSequenceRef.current = [...recentFeaturesRef.current]; 
        } else {
          recordedSequenceRef.current.push(feat);
          if (recordedSequenceRef.current.length > 150) recordedSequenceRef.current.shift();
        }
      } else {
        setCameraStatus("Diam");
        if (isSigningRef.current) {
          isSigningRef.current = false;
          
          if (recordedSequenceRef.current.length >= 10) {
            isPredictingRef.current = true;
            const finalSequence = processSequence(recordedSequenceRef.current, MAX_FRAMES);
            
            try {
              const inputTensor = tf.tensor3d([finalSequence]);
              const prediction = modelRef.current?.predict(inputTensor) as tf.Tensor;
              const scores = await prediction.data();
              
              const maxScore = Math.max(...Array.from(scores));
              const maxIndex = scores.indexOf(maxScore);
              tf.dispose([inputTensor, prediction]);

              if (maxScore > CONFIDENCE_THRESHOLD) {
                const newWord = classNamesRef.current[maxIndex].toUpperCase();
                setSentence(prev => [...prev, newWord]);
                
                if (currentIsAutoSpeak) {
                  speakText(newWord);
                }
              }
            } catch (err) {
              console.error("Error saat prediksi AI:", err);
            } finally {
              setTimeout(() => { isPredictingRef.current = false; }, 300);
            }
          }
          recordedSequenceRef.current = [];
        }
      }
    }
  };

  const processVideoFrame = async () => {
    if (videoRef.current && videoRef.current.readyState >= 2 && holisticRef.current) {
      await holisticRef.current.send({ image: videoRef.current });
    }
    if (videoRef.current && videoRef.current.srcObject) {
      animationFrameId.current = requestAnimationFrame(processVideoFrame);
    }
  };

  // ==========================================
  // FUNGSI HARDWARE KONTROL (KAMERA & MIC)
  // ==========================================

  const startCamera = async (mode = facingMode) => {
    try {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: mode } 
      });
      
      if (videoRef.current) videoRef.current.srcObject = stream;
      
      setIsCameraActive(true);
      recentFeaturesRef.current = [];
      recentRawHandsRef.current = []; 
      recordedSequenceRef.current = [];
      isSigningRef.current = false;
      
      processVideoFrame();
    } catch (err) {
      console.error("Akses kamera ditolak:", err);
      alert("Gagal mengakses kamera. Pastikan browser memiliki izin.");
      setIsCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setCameraStatus("Kamera Mati");
  };

  const toggleCameraMode = () => {
    const newMode = facingMode === "user" ? "environment" : "user";
    setFacingMode(newMode);
    if (isCameraActive) {
      stopCamera();
      setTimeout(() => startCamera(newMode), 300); 
    }
  };

  const startListening = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Browser Anda belum mendukung fitur Voice-to-Text.");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "id-ID"; 
    recognition.interimResults = true; 
    recognition.onstart = () => setIsListening(true);
    recognition.onresult = (event: any) => {
      const current = event.resultIndex;
      const text = event.results[current][0].transcript;
      setTranscript(text.toUpperCase());
    };
    recognition.onend = () => setIsListening(false);
    recognition.start();
  };

  const handleClearText = () => setTranscript("");
  const handleStartFromTutorial = () => {
    localStorage.setItem("signtara_tutorial_seen", "true");
    setShowTutorial(false);
  };

  // ==========================================
  // RENDER UI
  // ==========================================

  if (isCheckingMemory) return <div className="min-h-screen bg-[#FCF9F5]"></div>;

  if (showTutorial) {
    return (
      <div className="flex flex-col min-h-screen bg-[#FCF9F5]">
        <div className="bg-white rounded-b-[2.5rem] pt-8 pb-6 px-6 shadow-sm flex items-center justify-center relative z-10">
          <Link href="/" className="absolute left-6 text-[#F97316] hover:opacity-70 transition-opacity"><ArrowLeft size={24} /></Link>
          <h1 className="text-2xl font-bold text-[#F97316] tracking-wide">Signtara</h1>
        </div>
        <div className="px-8 pt-8 pb-10 flex flex-col flex-grow animate-in fade-in zoom-in duration-300">
          <div className="mb-12 text-center">
            <h2 className="text-2xl font-bold text-[#5C3A21] mb-2">Cara Penggunaan</h2>
            <p className="text-sm text-gray-500">Ikuti 3 langkah mudah ini untuk hasil terbaik.</p>
          </div>
          <div className="space-y-10 flex-grow">
            <div className="flex items-center gap-6 bg-white/50 p-4 rounded-3xl"><div className="w-16 h-16 shrink-0 bg-[#FCEEE6] rounded-2xl flex items-center justify-center text-[#5C3A21]"><Hand size={30} /></div><div><h3 className="font-bold text-[#5C3A21] text-lg">Posisi Tangan</h3><p className="text-sm text-gray-500 leading-snug">Pegang perangkat dengan stabil dan pastikan objek berada di tengah bingkai layar.</p></div></div>
            <div className="flex items-center gap-6 bg-white/50 p-4 rounded-3xl"><div className="w-16 h-16 shrink-0 bg-[#EBF5EE] rounded-2xl flex items-center justify-center text-[#5C3A21]"><RefreshCw size={30} /></div><div><h3 className="font-bold text-[#5C3A21] text-lg">Pencahayaan</h3><p className="text-sm text-gray-500 leading-snug">Pastikan area cukup terang. Hindari cahaya yang membelakangi objek.</p></div></div>
            <div className="flex items-center gap-6 bg-white/50 p-4 rounded-3xl"><div className="w-16 h-16 shrink-0 bg-[#E6F3FA] rounded-2xl flex items-center justify-center text-[#5C3A21]"><CameraIcon size={30} /></div><div><h3 className="font-bold text-[#5C3A21] text-lg">Jarak Ideal</h3><p className="text-sm text-gray-500 leading-snug">Jaga jarak sekitar 30-50 cm dari objek agar fokus kamera bekerja sempurna.</p></div></div>
          </div>
          <button onClick={handleStartFromTutorial} className="w-full bg-[#FFB18B] text-[#5C3A21] py-5 rounded-full font-bold text-lg shadow-sm hover:bg-[#EAA17B] transition-all flex items-center justify-center gap-3 mt-6 active:scale-95">
            Mengerti, Mulai Kamera <CameraIcon size={22} />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen pb-24 relative bg-[#FCF9F5]">
      
      {/* MODAL RIWAYAT (HISTORY) */}
      {showHistory && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/20 backdrop-blur-sm animate-in fade-in">
          <div className="w-full max-w-sm h-full bg-[#FCF9F5] shadow-2xl flex flex-col animate-in slide-in-from-right-8 duration-300">
            <div className="bg-white px-6 py-6 shadow-sm flex items-center justify-between">
              <h2 className="text-lg font-bold text-[#5C3A21] flex items-center gap-2"><History size={20}/> Riwayat</h2>
              <button onClick={() => setShowHistory(false)} className="text-gray-400 hover:text-red-500 transition-colors p-2"><X size={24}/></button>
            </div>
            <div className="flex-grow overflow-y-auto p-6 space-y-4">
              {history.length === 0 ? (
                 <p className="text-center text-gray-400 text-sm mt-10">Belum ada riwayat tersimpan.</p>
              ) : (
                 history.map(item => (
                   <div key={item.id} className="bg-white p-4 rounded-2xl shadow-sm border border-orange-50 relative group">
                      <div className="flex items-center gap-2 mb-2">
                         <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${item.type === 'tuli' ? 'bg-orange-100 text-orange-600' : 'bg-blue-100 text-blue-600'}`}>
                           {item.type === 'tuli' ? 'Isyarat' : 'Suara'}
                         </span>
                         <span className="text-[10px] text-gray-400">
                           {new Date(item.timestamp).toLocaleTimeString('id-ID', {hour: '2-digit', minute:'2-digit'})}
                         </span>
                      </div>
                      <p className="text-sm font-bold text-[#5C3A21] pr-8">{item.text}</p>
                      
                      <div className="absolute top-4 right-4 flex gap-1 opacity-100 md:opacity-0 group-hover:opacity-100 transition-opacity">
                        <button onClick={() => speakText(item.text)} className="text-gray-300 hover:text-green-500 bg-gray-50 p-1.5 rounded-full" title="Bunyikan">
                          <Volume2 size={14}/>
                        </button>
                        <button onClick={() => setHistory(prev => prev.filter(h => h.id !== item.id))} className="text-gray-300 hover:text-red-500 bg-gray-50 p-1.5 rounded-full" title="Hapus">
                          <Trash2 size={14}/>
                        </button>
                      </div>
                   </div>
                 ))
              )}
            </div>
            {history.length > 0 && (
              <div className="p-6 bg-white border-t border-gray-100">
                <button onClick={() => setHistory([])} className="w-full py-3 bg-red-50 text-red-500 font-bold rounded-xl hover:bg-red-100 transition-colors flex items-center justify-center gap-2 active:scale-95">
                  <Trash2 size={18}/> Hapus Semua
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* HEADER */}
      <div className="bg-white rounded-b-[2.5rem] pt-8 pb-6 px-6 shadow-sm flex items-center justify-center relative z-10 shrink-0">
        <Link href="/" className="absolute left-6 text-[#F97316] hover:opacity-70 transition-opacity"><ArrowLeft size={24} /></Link>
        <h1 className="text-2xl font-bold text-[#F97316] tracking-wide">Signtara</h1>
        
        <div className="absolute right-6 flex items-center gap-3">
          <button onClick={() => setShowHistory(true)} className="text-[#F97316] hover:opacity-70 transition-opacity"><History size={24} /></button>
          <button onClick={() => setShowTutorial(true)} className="text-[#F97316] hover:opacity-70 transition-opacity"><HelpCircle size={24} /></button>
        </div>
      </div>

      <div className="px-6 pt-8 flex-grow flex flex-col relative z-10 animate-in fade-in duration-500">
        
        {/* TABS BUTTON */}
        <div className="bg-gray-200/50 p-1 rounded-full flex mb-8 shrink-0">
          <button onClick={() => setActiveTab("tuli")} className={`flex-1 py-3.5 rounded-full font-bold text-sm transition-all ${activeTab === 'tuli' ? 'bg-[#5C3A21] text-white shadow-md' : 'text-gray-500'}`}>Teman Tuli</button>
          <button onClick={() => setActiveTab("dengar")} className={`flex-1 py-3.5 rounded-full font-bold text-sm transition-all ${activeTab === 'dengar' ? 'bg-[#5C3A21] text-white shadow-md' : 'text-gray-500'}`}>Teman Dengar</button>
        </div>

        <div className="flex flex-col flex-grow w-full relative z-10 animate-in fade-in duration-300">
          {activeTab === "tuli" ? (
            <div className="flex flex-col h-full animate-in slide-in-from-left-4 duration-300">
              
              {/* KOTAK KAMERA (TEMAN TULI) */}
              <div className="relative w-full aspect-[4/5] bg-gray-300 rounded-[3rem] overflow-hidden border-4 border-white shadow-xl mb-6 flex flex-col items-center justify-center bg-black shrink-0">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted 
                  className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-500 ${facingMode === 'user' ? 'scale-x-[-1]' : ''} ${isCameraActive ? 'opacity-100' : 'opacity-0'}`} 
                />

                {isCameraActive && (
                  <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center p-6 sm:p-10 opacity-60">
                    <div className="w-full h-full border-2 border-dashed border-white/70 rounded-[2rem] flex flex-col items-center pt-8 pb-4 relative">
                      <div className="w-20 h-28 border-2 border-dashed border-white/50 rounded-[40%] absolute top-8"></div>
                      <p className="mt-auto text-white text-[10px] sm:text-xs font-bold tracking-widest bg-black/30 px-3 py-1 rounded-full backdrop-blur-sm text-center max-w-[80%]">
                        Posisikan setengah badan di dalam garis
                      </p>
                    </div>
                  </div>
                )}

                {!isCameraActive ? (
                  <button onClick={() => startCamera(facingMode)} className="flex flex-col items-center gap-3 group z-20">
                    <div className="w-16 h-16 bg-white/80 rounded-full flex items-center justify-center text-[#F97316] group-hover:scale-110 transition-transform shadow-lg"><Play size={32} className="ml-1" /></div>
                    <p className="text-sm font-bold text-[#5C3A21] bg-white/80 px-4 py-1.5 rounded-full backdrop-blur-sm shadow-sm">Tap untuk Menyalakan Kamera</p>
                  </button>
                ) : (
                  <>
                    <div className="absolute top-4 right-4 flex gap-2 z-20">
                      <button onClick={toggleCameraMode} className="bg-white/80 p-3 rounded-full text-[#F97316] shadow-sm hover:bg-white transition-colors" title="Balik Kamera">
                        <SwitchCamera size={20} />
                      </button>
                      <button onClick={stopCamera} className="bg-white/80 p-3 rounded-full text-red-500 shadow-sm hover:bg-white transition-colors" title="Matikan Kamera">
                        <Square size={20} fill="currentColor" />
                      </button>
                    </div>
                    
                    <div className={`absolute top-4 left-4 backdrop-blur-md px-3 py-1.5 rounded-xl border flex items-center gap-2 shadow-sm z-20 transition-colors ${cameraStatus === 'Merekam...' ? 'bg-orange-500/80 border-orange-400' : 'bg-black/50 border-gray-500'}`}>
                      <div className={`w-2 h-2 rounded-full ${cameraStatus === 'Merekam...' ? 'bg-white animate-pulse' : 'bg-gray-300'}`} />
                      <span className="text-[11px] font-bold text-white tracking-wide">{cameraStatus}</span>
                    </div>

                    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-white/80 backdrop-blur-md px-4 py-2 rounded-full border border-white flex items-center gap-2 shadow-sm z-20">
                      <div className="w-2 h-2 rounded-full animate-pulse bg-[#F97316]" /><span className="text-[10px] font-bold uppercase tracking-wider text-[#5C3A21]">Teman Tuli Mode</span>
                    </div>
                  </>
                )}
              </div>

              {/* PANEL TERJEMAHAN (TEMAN TULI) */}
              <div className="bg-white rounded-[2.5rem] p-6 shadow-lg flex flex-col relative min-h-[220px] border border-orange-50 mt-auto mb-2 shrink-0">
                <div className="w-full flex justify-between items-center mb-4 shrink-0">
                  <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Kalimat Terdeteksi</p>
                  
                  {sentence.length > 0 && (
                    <div className="flex gap-2">
                      <button onClick={() => handleSaveHistory(sentence.join(" "), "tuli")} className="p-2 bg-blue-50 text-blue-500 rounded-full hover:bg-blue-100 transition-colors active:scale-95" title="Simpan ke Riwayat">
                        <Save size={18} />
                      </button>
                      <div className="w-px h-6 bg-gray-200 mx-1"></div>
                      <button onClick={handleUndo} className="p-2 bg-red-50 text-red-500 rounded-full hover:bg-red-100 transition-colors active:scale-95" title="Hapus Kata Terakhir">
                        <Delete size={18} />
                      </button>
                      <button onClick={handleClearSentence} className="p-2 bg-gray-50 text-gray-500 rounded-full hover:bg-gray-100 transition-colors active:scale-95" title="Hapus Semua">
                        <X size={18} />
                      </button>
                    </div>
                  )}
                </div>

                <div className="flex-grow w-full overflow-y-auto max-h-32 flex items-center justify-center mb-6">
                  <h2 
                    className="font-black text-[#5C3A21] text-center transition-all duration-100"
                    style={{ fontSize: `${16 + (textSize / 100) * 32}px`, lineHeight: '1.2' }}
                  >
                    {!isModelReady ? (
                      <span className="flex items-center gap-2 animate-pulse text-gray-400 text-xl justify-center">
                        <RefreshCw className="animate-spin" size={20}/> MENYIAPKAN AI...
                      </span>
                    ) : sentence.length > 0 ? (
                      sentence.join(" ") 
                    ) : (
                      <span className="text-gray-300 text-2xl">MENUNGGU ISYARAT...</span>
                    )}
                  </h2>
                </div>

                <div className="w-full flex flex-col gap-4 mt-auto shrink-0">
                  <div className="flex items-center gap-3 w-full px-2">
                    <span className="text-xs font-bold text-gray-300">T</span>
                    <input 
                      type="range" 
                      min="1" 
                      max="100" 
                      value={textSize}
                      onChange={(e) => setTextSize(Number(e.target.value))}
                      className="flex-grow h-1.5 bg-gray-100 rounded-full appearance-none accent-[#F97316]" 
                    />
                    <span className="text-lg font-bold text-gray-400">T</span>
                  </div>

                  <div className="w-full flex items-center justify-between gap-3">
                    <button 
                      onClick={() => setIsAutoSpeak(!isAutoSpeak)}
                      className={`p-3.5 rounded-2xl transition-colors active:scale-95 ${isAutoSpeak ? 'bg-green-500 text-white shadow-md' : 'bg-[#EBF5EE] text-green-600 hover:bg-green-100'}`}
                      title={isAutoSpeak ? "Matikan Suara Otomatis" : "Nyalakan Suara Otomatis per Kata"}
                    >
                      <Volume2 size={20} />
                    </button>
                    <button 
                      onClick={speakFullSentence}
                      className="flex-grow py-3.5 bg-[#F97316] text-white rounded-2xl font-bold shadow-md hover:bg-orange-600 transition-colors active:scale-95 flex items-center justify-center gap-2"
                    >
                      <Play size={18} fill="currentColor" /> Bunyikan Kalimat
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col h-full animate-in slide-in-from-right-4 duration-300">
              
              {/* PANEL SUARA (TEMAN DENGAR) */}
              <div className="bg-white rounded-[2.5rem] p-6 sm:p-8 shadow-md flex flex-col flex-1 items-center relative border border-orange-50 mb-6 overflow-hidden">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-4 shrink-0">Detected Voice</p>
                
                {transcript.length > 0 && (
                  <div className="absolute top-6 right-6 flex gap-2 z-10 shrink-0">
                    <button onClick={() => handleSaveHistory(transcript, "dengar")} className="text-gray-300 hover:text-blue-500 transition-colors p-2 bg-gray-50 rounded-full active:scale-95" title="Simpan ke Riwayat">
                      <Save size={20} />
                    </button>
                    <button onClick={handleClearText} className="text-gray-300 hover:text-red-500 transition-colors p-2 bg-gray-50 rounded-full active:scale-95" title="Hapus Teks">
                      <X size={20} />
                    </button>
                  </div>
                )}

                <div className="w-full flex-grow overflow-y-auto flex flex-col justify-center px-2 pb-2 mt-4">
                  {isTyping ? (
                    <textarea 
                      autoFocus
                      value={transcript}
                      onChange={(e) => setTranscript(e.target.value.toUpperCase())}
                      placeholder="Ketik pesan di sini..."
                      className="w-full h-full min-h-[150px] bg-gray-50 rounded-2xl text-[#5C3A21] font-bold text-xl focus:outline-none focus:ring-2 focus:ring-[#F97316] transition-all resize-none p-4 break-words"
                    />
                  ) : (
                    <h2 className="text-2xl font-bold text-[#5C3A21] text-center leading-relaxed m-auto w-full break-words break-all sm:break-normal whitespace-pre-wrap">
                      {transcript || <span className="text-gray-300 font-medium text-xl">Silakan tekan mikrofon untuk berbicara...</span>}
                    </h2>
                  )}
                </div>
              </div>

              <div className="bg-white rounded-[2.5rem] p-6 shadow-md flex items-center gap-6 border border-orange-50 relative shrink-0">
                {isTyping ? (
                  <div className="w-full flex gap-3">
                    <button onClick={() => setIsTyping(false)} className="flex-1 py-4 bg-gray-100 rounded-xl font-bold text-gray-500">Tutup</button>
                    <button onClick={() => setIsTyping(false)} className="flex-1 py-4 bg-[#F97316] text-white rounded-xl font-bold">Selesai</button>
                  </div>
                ) : (
                  <>
                    <button 
                      onClick={startListening}
                      className={`w-16 h-16 rounded-full flex items-center justify-center shrink-0 shadow-inner transition-colors ${isListening ? 'bg-orange-200' : 'bg-[#FCEEE6]'}`}
                    >
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center text-white transition-all ${isListening ? 'bg-red-500 animate-pulse scale-110' : 'bg-[#F97316]'}`}>
                        <Mic size={24} />
                      </div>
                    </button>
                    <div className="flex flex-col">
                      <p className="font-bold text-[#F97316] text-sm">{isListening ? 'Mendengarkan...' : 'Tekan Mic'}</p>
                      <p className="text-xs text-gray-400">{isListening ? 'Silakan berbicara sekarang.' : 'Untuk mulai mengubah suara.'}</p>
                    </div>
                    <button onClick={() => setIsTyping(true)} className="absolute top-6 right-6 text-gray-300 hover:text-[#F97316]">
                      <Keyboard size={20} />
                    </button>
                  </>
                )}
              </div>
              <div className="mx-auto bg-white/80 backdrop-blur-md px-5 py-2.5 rounded-full border border-gray-100 flex items-center gap-2 shadow-sm mt-8 pb-2 shrink-0">
                <div className="w-2 h-2 rounded-full animate-pulse bg-blue-500" />
                <span className="text-[10px] font-bold uppercase tracking-wider text-[#5C3A21]">Teman Dengar Mode</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}