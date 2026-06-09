"use client";

import { useState, useEffect } from "react";
import { ArrowLeft, Clock, Trash2, Search, Volume2, Copy, Check, Trash } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

interface HistoryItem {
  id: number;
  text: string;
  mode: string;
  timestamp: string;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [copiedId, setCopiedId] = useState<number | null>(null);
  const router = useRouter();

  useEffect(() => {
    const savedHistory = localStorage.getItem("signtara_history");
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  const clearHistory = () => {
    if (window.confirm("Apakah Anda yakin ingin menghapus semua riwayat percakapan?")) {
      localStorage.removeItem("signtara_history");
      setHistory([]);
      alert("Semua riwayat berhasil dihapus!");
    }
  };

  const deleteItem = (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    if (window.confirm("Hapus pesan ini dari riwayat?")) {
      const updatedHistory = history.filter(item => item.id !== id);
      setHistory(updatedHistory);
      localStorage.setItem("signtara_history", JSON.stringify(updatedHistory));
      alert("Pesan dihapus!");
    }
  };

  const copyToClipboard = (e: React.MouseEvent, id: number, text: string) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    alert("Teks berhasil disalin ke papan klip!");
    setTimeout(() => setCopiedId(null), 2000);
  };

  const speakText = (e: React.MouseEvent, text: string) => {
    e.stopPropagation();
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); 
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'id-ID';
      utterance.rate = 0.9;
      window.speechSynthesis.speak(utterance);
    } else {
      alert("Maaf, browser Anda tidak mendukung fitur Text-to-Speech.");
    }
  };

  const handleReload = (text: string, mode: string) => {
    localStorage.setItem("signtara_reload_text", text);
    localStorage.setItem("signtara_reload_mode", mode);
    
    router.push("/translate");
  };

  const filteredHistory = history.filter(item => 
    item.text.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-[#FCF9F5] flex flex-col">
      <div className="bg-white rounded-b-[2.5rem] pt-8 pb-6 px-6 shadow-sm flex items-center justify-between relative z-10 shrink-0">
        <Link href="/" className="text-[#F97316] hover:opacity-70 transition-opacity">
          <ArrowLeft size={24} />
        </Link>
        <h1 className="text-xl font-bold text-[#5C3A21]">Riwayat Obrolan</h1>
        <button 
          onClick={clearHistory} 
          className="text-red-400 hover:text-red-600 transition-colors disabled:opacity-50"
          disabled={history.length === 0}
          title="Hapus Semua Riwayat"
        >
          <Trash2 size={24} />
        </button>
      </div>

      <div className="px-6 pt-6 flex-grow flex flex-col animate-in fade-in duration-500 pb-24">
        
        <div className="bg-white rounded-full flex items-center px-4 py-3 shadow-sm border border-orange-50 mb-6 shrink-0">
          <Search size={20} className="text-gray-400 mr-3" />
          <input 
            type="text" 
            placeholder="Cari percakapan..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="flex-grow bg-transparent focus:outline-none text-[#5C3A21] font-medium placeholder:text-gray-300"
          />
        </div>

        <div className="flex flex-col gap-4 overflow-y-auto">
          {filteredHistory.length === 0 ? (
            <div className="flex flex-col items-center justify-center mt-20 text-center opacity-50">
              <Clock size={48} className="text-[#F97316] mb-4" />
              <p className="text-[#5C3A21] font-bold text-lg">Belum ada riwayat</p>
              <p className="text-gray-400 text-sm mt-1">Percakapan yang Anda simpan akan muncul di sini.</p>
            </div>
          ) : (
            filteredHistory.map((item) => (
              <div 
                key={item.id} 
                onClick={() => handleReload(item.text, item.mode)}
                className="bg-white p-5 rounded-3xl shadow-sm border border-orange-50 relative overflow-hidden group cursor-pointer hover:shadow-md transition-shadow"
                title="Tekan untuk menampilkan kembali di halaman Translate"
              >
                <div className={`absolute top-0 left-0 w-1.5 h-full ${item.mode === 'Teman Tuli' ? 'bg-[#F97316]' : 'bg-blue-400'}`}></div>
                
                <div className="flex justify-between items-start mb-2 pl-2">
                  <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-full ${item.mode === 'Teman Tuli' ? 'bg-[#FCEEE6] text-[#F97316]' : 'bg-blue-50 text-blue-600'}`}>
                    {item.mode}
                  </span>
                  <span className="text-xs text-gray-400 font-medium">{item.timestamp}</span>
                </div>
                
                <p className="text-lg font-bold text-[#5C3A21] pl-2 mt-2 leading-snug break-words">
                  {item.text}
                </p>

                <div className="mt-4 pt-3 border-t border-gray-100 flex justify-end gap-2 pl-2">
                  <button 
                    onClick={(e) => speakText(e, item.text)}
                    className="p-2 bg-[#EBF5EE] text-green-600 rounded-xl hover:bg-green-100 transition-colors active:scale-95"
                    title="Putar Suara"
                  >
                    <Volume2 size={18} />
                  </button>
                  <button 
                    onClick={(e) => copyToClipboard(e, item.id, item.text)}
                    className={`p-2 rounded-xl transition-colors active:scale-95 ${copiedId === item.id ? 'bg-blue-500 text-white' : 'bg-[#E6F3FA] text-blue-600 hover:bg-blue-100'}`}
                    title="Salin Teks"
                  >
                    {copiedId === item.id ? <Check size={18} /> : <Copy size={18} />}
                  </button>
                  <button 
                    onClick={(e) => deleteItem(e, item.id)}
                    className="p-2 bg-red-50 text-red-500 rounded-xl hover:bg-red-100 transition-colors active:scale-95"
                    title="Hapus"
                  >
                    <Trash size={18} />
                  </button>
                </div>

              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}