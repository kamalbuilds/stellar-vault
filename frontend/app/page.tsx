import { Hero } from '@/components/sections/hero';
import { Features } from '@/components/sections/features';
import { AssetTypes } from '@/components/sections/asset-types';
import { Stats } from '@/components/sections/stats';
import { TechStack } from '@/components/sections/tech-stack';
import { CTA } from '@/components/sections/cta';

export default function HomePage() {
  return (
    <div className="min-h-screen">
      <Hero />
      <Stats />
      <Features />
      <AssetTypes />
      <TechStack />
      <CTA />
    </div>
  );
} 