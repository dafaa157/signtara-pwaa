"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Camera, History, Info } from "lucide-react";

export default function BottomNav() {
  const pathname = usePathname();

  const navItems = [
    { name: "Beranda", path: "/", icon: Home },
    { name: "Terjemahan", path: "/translate", icon: Camera },
    { name: "Riwayat", path: "/history", icon: History },
    { name: "Tentang", path: "/about", icon: Info },
  ];

  return (
    <div className="fixed bottom-6 left-0 w-full flex justify-center z-50">
      <nav className="bg-white px-6 py-3 rounded-full shadow-[0_8px_30px_rgb(0,0,0,0.08)] flex gap-8 items-center border border-gray-100">
        {navItems.map((item) => {
          const isActive = pathname === item.path;
          const Icon = item.icon;

          return (
            <Link href={item.path} key={item.name} className="flex flex-col items-center gap-1">
              <div
                className={`p-2 rounded-full transition-colors ${
                  isActive ? "bg-signtara-orange/20 text-signtara-orange" : "text-gray-400 hover:text-gray-600"
                }`}
              >
                <Icon size={24} strokeWidth={isActive ? 2.5 : 2} />
              </div>
              <span
                className={`text-[10px] font-medium ${
                  isActive ? "text-signtara-orange" : "text-gray-400"
                }`}
              >
                {item.name}
              </span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}