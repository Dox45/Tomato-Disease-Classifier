import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    // ❌ Don’t run ESLint during builds
    ignoreDuringBuilds: true,
  },
  typescript: {
    // ❌ Don’t block production builds on TS errors
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
