import { ArrowRight, Zap, ShieldCheck, Heart } from "lucide-react";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen pb-24 relative bg-[#FCF9F5]">
      <div className="bg-white rounded-b-[2.5rem] pt-6 pb-6 px-6 shadow-sm flex justify-center items-center z-10 relative">
        <h1 className="text-2xl font-bold text-[#F97316] tracking-wide">Signtara</h1>
      </div>

      <div className="px-6 pt-8 animate-in fade-in duration-500">
        <div className="relative mb-10 mt-2">
          <div className="absolute -top-4 -right-2 w-32 h-32 bg-[#FDEADD] rounded-full blur-2xl opacity-70"></div>
          <div className="absolute -bottom-4 -left-2 w-32 h-32 bg-[#E6F4EA] rounded-full blur-2xl opacity-70"></div>
          
          <div className="relative bg-white/60 backdrop-blur-md rounded-[2.5rem] p-8 shadow-sm border border-white flex flex-col items-center text-center">
            <div className="bg-[#EBB49E] w-16 h-16 rounded-full flex items-center justify-center mb-5 text-white shadow-inner overflow-hidden border border-gray-400">
              <img 
                src="/logo.png" 
                alt="Logo Signtara" 
                className="w-full h-full object-cover scale-110" 
              />
            </div>
            <h2 className="text-xl font-bold text-[#5C3A21] mb-2">Jembatan Komunikasi</h2>
            <p className="text-sm text-[#5C3A21] mb-8 font-medium">
              Terhubung dengan lancar melalui terjemahan instan dan akurat.
            </p>
            
            <Link 
              href="/translate" 
              className="bg-[#6B4C3A] text-white w-full py-4 rounded-full font-bold flex items-center justify-center gap-2 hover:bg-[#5C3A21] transition-colors shadow-md"
            >
              Mulai Terjemahkan <ArrowRight size={20} />
            </Link>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-bold text-[#5C3A21] mb-4 ml-1">Kenapa Signtara?</h3>
          
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-[#D1EAD8] rounded-[2rem] p-6 shadow-sm flex flex-col gap-8">
              <div className="bg-white/60 w-10 h-10 rounded-full flex items-center justify-center text-[#5C3A21]">
                <Zap size={20} />
              </div>
              <p className="font-bold text-[#5C3A21] text-sm leading-tight">
                Kecepatan<br/>Real-time
              </p>
            </div>

            <div className="bg-[#93C5E6] rounded-[2rem] p-6 shadow-sm flex flex-col gap-8">
              <div className="bg-white/60 w-10 h-10 rounded-full flex items-center justify-center text-[#5C3A21]">
                <ShieldCheck size={20} />
              </div>
              <p className="font-bold text-[#5C3A21] text-sm leading-tight">
                Akurasi<br/>Tinggi
              </p>
            </div>
          </div>

          <div className="bg-[#FFE5D4] rounded-[2rem] p-6 shadow-sm flex items-start gap-4">
            <div className="bg-white/60 min-w-[2.5rem] h-10 rounded-full flex items-center justify-center text-[#5C3A21] mt-1">
              <Heart size={20} />
            </div>
            <div>
              <h4 className="font-bold text-[#5C3A21] text-sm mb-1">Komunitas Inklusif</h4>
              <p className="text-xs text-[#5C3A21] font-medium leading-relaxed">
                Dibangun untuk membuat komunikasi dapat diakses oleh semua orang.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}