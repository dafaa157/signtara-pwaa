import { ArrowLeft, Users, Hand, Layers } from "lucide-react";
import Link from "next/link";

export default function AboutPage() {
  const teamMembers = [
    { name: "Ryan", url: "/ryan.jpeg" },
    { name: "Tangkas", url: "/tangkas.jpeg" },
    { name: "Fatoni", url: "/fatoni.jpeg" },
    { name: "Daffa", url: "/daffa.jpeg" },
  ];

  return (
    <div className="flex flex-col min-h-screen pb-24 relative bg-[#FCF9F5]">
      
      <div className="bg-white rounded-b-[2.5rem] pt-8 pb-6 px-6 shadow-sm flex items-center justify-center relative z-10">
        <Link href="/" className="absolute left-6 text-[#F97316] hover:opacity-70 transition-opacity">
          <ArrowLeft size={24} />
        </Link>
        <h1 className="text-2xl font-bold text-[#F97316] tracking-wide">Signtara</h1>
      </div>

      <div className="px-6 pt-8 animate-in fade-in duration-500">
        
        <div className="bg-[#EBB49E] rounded-[2.5rem] p-8 mb-8 shadow-sm">
          <div className="bg-white/50 w-12 h-12 rounded-full flex items-center justify-center mb-5 text-[#5C3A21]">
            <Users size={24} />
          </div>
          <h2 className="text-xl font-bold text-[#5C3A21] mb-3">Visi Kami</h2>
          <p className="text-sm text-[#5C3A21] font-medium leading-relaxed">
            Membangun jembatan komunikasi yang inklusif untuk mematahkan isolasi sosial. Kami percaya setiap <b>Teman Tuli</b> berhak untuk terhubung, berekspresi, dan dipahami dalam kehidupan sehari-hari dengan nyaman dan aman.
          </p>
        </div>

        <div className="mb-8">
          <h2 className="text-lg font-bold text-[#5C3A21] mb-4 ml-1">Transparansi Teknologi</h2>
          <div className="bg-[#D1EAD8] rounded-[2.5rem] p-6 shadow-sm">
            <p className="text-sm text-[#5C3A21] font-medium mb-6 leading-relaxed px-2">
              Privasi Anda adalah prioritas. Signtara dirancang untuk memproses data bahasa isyarat secara lokal di perangkat Anda.
            </p>

            <div className="bg-white rounded-[1.5rem] p-5 mb-4 shadow-sm flex flex-col gap-2">
              <div className="flex items-center gap-3">
                <Hand size={20} className="text-[#5C3A21]" />
                <h3 className="font-bold text-[#5C3A21]">MediaPipe</h3>
              </div>
              <p className="text-xs text-[#5C3A21] font-medium leading-relaxed">
                Melacak titik presisi gerakan tangan secara real-time langsung di smartphone.
              </p>
            </div>

            <div className="bg-white rounded-[1.5rem] p-5 shadow-sm flex flex-col gap-2">
              <div className="flex items-center gap-3">
                <Layers size={20} className="text-[#5C3A21]" />
                <h3 className="font-bold text-[#5C3A21]">CNN 1D</h3>
              </div>
              <p className="text-xs text-[#5C3A21] font-medium leading-relaxed">
                CNN 1D (Convolutional Neural Network) untuk mengekstraksi pola fitur spasial dari urutan gerakan secara cepat dan akurat.
              </p>
            </div>
          </div>
        </div>

        <div>
          <h2 className="text-lg font-bold text-[#5C3A21] mb-4 ml-1">Tim Pengembang</h2>
          <div className="grid grid-cols-2 gap-4">
            {teamMembers.map((member, index) => (
              <div key={index} className="bg-white rounded-[2.5rem] p-3 pb-5 shadow-sm border border-gray-50 flex flex-col items-center">
                <div className="w-full aspect-square rounded-full bg-[#FCF9F5] mb-3 flex items-center justify-center shadow-inner border border-gray-100">
                  <img src={member.url} className="w-full h-full object-cover rounded-full" />
                </div>
                <p className="font-bold text-[#5C3A21]">{member.name}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}