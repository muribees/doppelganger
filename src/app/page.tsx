"use client";

import { useState, useRef, useEffect, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Celebrity {
  name: string;
  file: string;
}

interface MatchResult {
  name: string;
  similarity: number;
  image: string;
}

type Stage =
  | "hero"
  | "loading-models"
  | "ready"
  | "capturing"
  | "processing"
  | "results";

/* ------------------------------------------------------------------ */
/*  Celebrity list                                                     */
/* ------------------------------------------------------------------ */

const CELEBRITIES: Celebrity[] = [
  { name: "Audrey Hepburn", file: "audrey_hepburn.jpg" },
  { name: "Bad Bunny", file: "bad_bunny.jpg" },
  { name: "Donald Trump", file: "donald_trump.jpg" },
  { name: "Jenna Ortega", file: "jenna_ortega.jpg" },
  { name: "Ken Jeong", file: "ken_jeong.jpg" },
  { name: "Michael Jackson", file: "michael_jackson.jpg" },
  { name: "Michelle Yeoh", file: "michelle_yeoh.jpg" },
  { name: "RDJ", file: "rdj.jpg" },
  { name: "Rachel McAdams", file: "rachel_mcadams.jpg" },
  { name: "Ryan Gosling", file: "ryan_gosling.jpg" },
  { name: "Suni Lee", file: "suni_lee.jpg" },
  { name: "Timothée Chalamet", file: "timothee_chalamet.jpg" },
  { name: "Tom Hiddleston", file: "tom_hiddleston.jpg" },
  { name: "Zayn Malik", file: "zayn_malik.jpg" },
  { name: "Zendaya", file: "zendaya.jpg" },
];

/* ------------------------------------------------------------------ */
/*  face-api.js helpers (loaded dynamically)                           */
/* ------------------------------------------------------------------ */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let faceapi: any = null;

const MODEL_URL =
  "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights";

async function loadFaceApi() {
  if (faceapi) return faceapi;
  const mod = await import(
    // @ts-expect-error - no types for CDN module
    /* webpackIgnore: true */ "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"
  ).catch(() => null);

  if (mod) {
    faceapi = mod;
  } else {
    // fallback: load via script tag
    await new Promise<void>((resolve, reject) => {
      const s = document.createElement("script");
      s.src =
        "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js";
      s.onload = () => resolve();
      s.onerror = reject;
      document.head.appendChild(s);
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    faceapi = (window as any).faceapi;
  }

  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  ]);

  return faceapi;
}

async function getDescriptor(
  input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
): Promise<Float32Array | null> {
  const detection = await faceapi
    .detectSingleFace(input)
    .withFaceLandmarks()
    .withFaceDescriptor();
  return detection ? detection.descriptor : null;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function Home() {
  const [stage, setStage] = useState<Stage>("hero");
  const [loadProgress, setLoadProgress] = useState("");
  const [results, setResults] = useState<MatchResult[]>([]);
  const [userPhoto, setUserPhoto] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const celebDescriptors = useRef<
    { name: string; descriptor: Float32Array; image: string }[]
  >([]);

  /* ---- Load models + celebrity embeddings ---- */
  const initialize = useCallback(async () => {
    setStage("loading-models");
    setError(null);
    try {
      setLoadProgress("Loading face recognition models...");
      await loadFaceApi();

      setLoadProgress("Processing celebrity faces...");
      const descriptors: typeof celebDescriptors.current = [];

      for (const celeb of CELEBRITIES) {
        setLoadProgress(`Scanning ${celeb.name}...`);
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = `/celebrities/${celeb.file}`;
        await new Promise<void>((res, rej) => {
          img.onload = () => res();
          img.onerror = rej;
        });

        const desc = await getDescriptor(img);
        if (desc) {
          descriptors.push({
            name: celeb.name,
            descriptor: desc,
            image: `/celebrities/${celeb.file}`,
          });
        }
      }

      celebDescriptors.current = descriptors;
      setStage("ready");
    } catch (e) {
      console.error(e);
      setError(
        "Failed to load face recognition models. Please refresh and try again."
      );
      setStage("hero");
    }
  }, []);

  /* ---- Webcam ---- */
  const startCamera = async () => {
    setStage("capturing");
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch {
      setError(
        "Camera access denied. Please allow camera access or upload a photo instead."
      );
      setStage("ready");
    }
  };

  const capturePhoto = async () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(video, 0, 0);

    // stop camera
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    setUserPhoto(dataUrl);
    await findMatch(canvas);
  };

  /* ---- Upload ---- */
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async () => {
      const dataUrl = reader.result as string;
      setUserPhoto(dataUrl);

      const img = new Image();
      img.src = dataUrl;
      await new Promise<void>((res) => (img.onload = () => res()));

      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      canvas.getContext("2d")!.drawImage(img, 0, 0);
      await findMatch(canvas);
    };
    reader.readAsDataURL(file);
  };

  /* ---- Match ---- */
  const findMatch = async (canvas: HTMLCanvasElement) => {
    setStage("processing");
    setError(null);

    const desc = await getDescriptor(canvas);
    if (!desc) {
      setError(
        "No face detected in your photo. Try again with better lighting and face the camera directly."
      );
      setStage("ready");
      return;
    }

    const scored = celebDescriptors.current.map((c) => ({
      name: c.name,
      similarity: cosineSimilarity(desc, c.descriptor),
      image: c.image,
    }));
    scored.sort((a, b) => b.similarity - a.similarity);

    setResults(scored);
    setStage("results");
  };

  /* ---- Reset ---- */
  const reset = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setUserPhoto(null);
    setResults([]);
    setStage("ready");
  };

  /* ---- Cleanup on unmount ---- */
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <main className="min-h-screen">
      {/* ===== HERO ===== */}
      <section className="relative flex flex-col items-center justify-center min-h-screen px-4 text-center overflow-hidden">
        {/* background glow */}
        <div
          className="absolute w-[600px] h-[600px] rounded-full opacity-20 blur-[120px] pointer-events-none"
          style={{ background: "var(--accent)", top: "-10%", left: "50%", transform: "translateX(-50%)" }}
        />

        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-4 animate-fade-in-up">
          Who&rsquo;s Your{" "}
          <span style={{ color: "var(--accent-light)" }}>Doppelganger</span>?
        </h1>
        <p
          className="text-lg md:text-xl max-w-2xl mb-8 animate-fade-in-up"
          style={{ color: "var(--text-muted)", animationDelay: "0.15s" }}
        >
          Discover which celebrity you look most like using real facial
          recognition AI — running entirely in your browser.
        </p>

        {stage === "hero" && (
          <button
            onClick={initialize}
            className="px-8 py-4 text-lg font-semibold rounded-xl transition-all duration-200 cursor-pointer animate-fade-in-up hover:scale-105 active:scale-95"
            style={{
              background: "var(--accent)",
              color: "#fff",
              animationDelay: "0.3s",
            }}
          >
            Get Started
          </button>
        )}

        {stage === "loading-models" && (
          <div className="flex flex-col items-center gap-3 animate-fade-in-up">
            <div className="w-8 h-8 border-3 border-t-transparent rounded-full animate-spin" style={{ borderColor: "var(--accent)", borderTopColor: "transparent" }} />
            <p style={{ color: "var(--text-muted)" }}>{loadProgress}</p>
          </div>
        )}

        {error && (
          <p className="mt-4 text-red-400 max-w-md">{error}</p>
        )}

        {/* scroll hint */}
        {stage === "hero" && (
          <div className="absolute bottom-8 animate-bounce" style={{ color: "var(--text-muted)" }}>
            <p className="text-sm mb-1">Learn how it works</p>
            <svg
              className="mx-auto w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </div>
        )}
      </section>

      {/* ===== HOW IT WORKS ===== */}
      <section className="max-w-4xl mx-auto px-4 py-20">
        <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">
          How Does a Computer Recognize a Face?
        </h2>

        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              step: "1",
              title: "Detect",
              desc: "A neural network locates faces in the image, finding eyes, nose, and jawline.",
              icon: "👁️",
            },
            {
              step: "2",
              title: "Embed",
              desc: "The face is converted into 128 numbers — a unique fingerprint capturing eye spacing, nose shape, and more.",
              icon: "🧬",
            },
            {
              step: "3",
              title: "Compare",
              desc: "We measure how close two sets of 128 numbers are using cosine similarity — 1 means identical, 0 means totally different.",
              icon: "📊",
            },
          ].map((item) => (
            <div
              key={item.step}
              className="rounded-2xl p-6 border"
              style={{
                background: "var(--surface)",
                borderColor: "var(--border)",
              }}
            >
              <div className="text-3xl mb-3">{item.icon}</div>
              <div
                className="text-xs font-bold uppercase tracking-widest mb-1"
                style={{ color: "var(--accent-light)" }}
              >
                Step {item.step}
              </div>
              <h3 className="text-xl font-bold mb-2">{item.title}</h3>
              <p style={{ color: "var(--text-muted)" }}>{item.desc}</p>
            </div>
          ))}
        </div>

        <div
          className="mt-10 rounded-2xl p-6 border text-center"
          style={{
            background: "var(--surface)",
            borderColor: "var(--border)",
          }}
        >
          <p className="font-mono text-sm mb-2" style={{ color: "var(--text-muted)" }}>
            Your face →
          </p>
          <p className="font-mono text-xs md:text-sm break-all" style={{ color: "var(--accent-light)" }}>
            [0.023, -0.145, 0.089, 0.234, -0.012, 0.178, ... 128 numbers total]
          </p>
          <p className="mt-3 text-sm" style={{ color: "var(--text-muted)" }}>
            This is the <strong>exact same technology</strong> used by Face ID,
            Instagram filters, and airport security.
          </p>
        </div>
      </section>

      {/* ===== CELEBRITY DATABASE ===== */}
      <section className="max-w-5xl mx-auto px-4 py-16">
        <h2 className="text-3xl md:text-4xl font-bold mb-3 text-center">
          The Comparison Database
        </h2>
        <p className="text-center mb-10" style={{ color: "var(--text-muted)" }}>
          15 public figures — the AI extracts 128 numbers from each face.
        </p>
        <div className="grid grid-cols-3 sm:grid-cols-5 gap-3">
          {CELEBRITIES.map((c) => (
            <div key={c.name} className="text-center">
              <div
                className="aspect-square rounded-xl overflow-hidden border mb-2"
                style={{ borderColor: "var(--border)" }}
              >
                <img
                  src={`/celebrities/${c.file}`}
                  alt={c.name}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
              </div>
              <p className="text-xs md:text-sm font-medium truncate">{c.name}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ===== INTERACTIVE SECTION ===== */}
      {(stage === "ready" ||
        stage === "capturing" ||
        stage === "processing" ||
        stage === "results") && (
        <section
          id="try-it"
          className="max-w-3xl mx-auto px-4 py-20"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-10 text-center">
            Find{" "}
            <span style={{ color: "var(--accent-light)" }}>Your</span>{" "}
            Doppelganger
          </h2>

          {/* --- READY: show camera/upload options --- */}
          {stage === "ready" && (
            <div className="flex flex-col items-center gap-4">
              <button
                onClick={startCamera}
                className="px-8 py-4 text-lg font-semibold rounded-xl transition-all duration-200 cursor-pointer hover:scale-105 active:scale-95"
                style={{ background: "var(--accent)", color: "#fff" }}
              >
                Use Camera
              </button>
              <span style={{ color: "var(--text-muted)" }}>or</span>
              <label
                className="px-8 py-4 text-lg font-semibold rounded-xl cursor-pointer transition-all duration-200 hover:scale-105 border"
                style={{
                  borderColor: "var(--accent)",
                  color: "var(--accent-light)",
                }}
              >
                Upload a Photo
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleUpload}
                />
              </label>
              {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
            </div>
          )}

          {/* --- CAPTURING: video feed --- */}
          {stage === "capturing" && (
            <div className="flex flex-col items-center gap-4">
              <div
                className="rounded-2xl overflow-hidden border"
                style={{ borderColor: "var(--accent)", borderWidth: 2 }}
              >
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="mirror w-full max-w-md"
                />
              </div>
              <button
                onClick={capturePhoto}
                className="px-8 py-4 text-lg font-semibold rounded-xl transition-all duration-200 cursor-pointer hover:scale-105 active:scale-95"
                style={{ background: "var(--green)", color: "#000" }}
              >
                Capture
              </button>
            </div>
          )}

          {/* --- PROCESSING --- */}
          {stage === "processing" && (
            <div className="flex flex-col items-center gap-4">
              {userPhoto && (
                <img
                  src={userPhoto}
                  alt="Your photo"
                  className="w-40 h-40 rounded-2xl object-cover border"
                  style={{ borderColor: "var(--accent)" }}
                />
              )}
              <div className="flex items-center gap-3">
                <div
                  className="w-6 h-6 border-3 border-t-transparent rounded-full animate-spin"
                  style={{
                    borderColor: "var(--accent)",
                    borderTopColor: "transparent",
                  }}
                />
                <p style={{ color: "var(--text-muted)" }}>
                  Analyzing your face...
                </p>
              </div>
            </div>
          )}

          {/* --- RESULTS --- */}
          {stage === "results" && results.length > 0 && (
            <div className="animate-fade-in-up">
              {/* Top match hero */}
              <div
                className="rounded-2xl p-8 border mb-8 text-center"
                style={{
                  background: "var(--surface)",
                  borderColor: "var(--accent)",
                  borderWidth: 2,
                }}
              >
                <p className="text-sm font-bold uppercase tracking-widest mb-4" style={{ color: "var(--accent-light)" }}>
                  Your Doppelganger
                </p>
                <div className="flex items-center justify-center gap-6 flex-wrap">
                  {userPhoto && (
                    <img
                      src={userPhoto}
                      alt="You"
                      className="w-32 h-32 md:w-40 md:h-40 rounded-2xl object-cover"
                    />
                  )}
                  <div className="text-4xl" style={{ color: "var(--accent-light)" }}>
                    ≈
                  </div>
                  <img
                    src={results[0].image}
                    alt={results[0].name}
                    className="w-32 h-32 md:w-40 md:h-40 rounded-2xl object-cover"
                  />
                </div>
                <h3 className="text-2xl md:text-3xl font-bold mt-6">
                  {results[0].name}
                </h3>
                <p className="text-lg mt-1" style={{ color: "var(--green)" }}>
                  {(results[0].similarity * 100).toFixed(1)}% match
                </p>
              </div>

              {/* Full ranking */}
              <div
                className="rounded-2xl border overflow-hidden"
                style={{
                  background: "var(--surface)",
                  borderColor: "var(--border)",
                }}
              >
                <div className="px-6 py-4 border-b" style={{ borderColor: "var(--border)" }}>
                  <h3 className="font-bold">Full Ranking</h3>
                </div>
                <div className="divide-y" style={{ borderColor: "var(--border)" }}>
                  {results.map((r, i) => (
                    <div
                      key={r.name}
                      className="flex items-center gap-4 px-6 py-3"
                      style={{ borderColor: "var(--border)" }}
                    >
                      <span
                        className="w-6 text-right text-sm font-bold"
                        style={{
                          color: i < 3 ? "var(--accent-light)" : "var(--text-muted)",
                        }}
                      >
                        {i + 1}
                      </span>
                      <img
                        src={r.image}
                        alt={r.name}
                        className="w-10 h-10 rounded-lg object-cover"
                      />
                      <span className="flex-1 font-medium text-sm">{r.name}</span>
                      <div className="w-24 md:w-40">
                        <div
                          className="h-2 rounded-full overflow-hidden"
                          style={{ background: "var(--border)" }}
                        >
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.max(r.similarity * 100, 2)}%`,
                              background:
                                i === 0
                                  ? "var(--green)"
                                  : i < 3
                                  ? "var(--accent)"
                                  : "var(--text-muted)",
                            }}
                          />
                        </div>
                      </div>
                      <span
                        className="text-sm w-14 text-right font-mono"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {(r.similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="text-center mt-8">
                <button
                  onClick={reset}
                  className="px-6 py-3 font-semibold rounded-xl transition-all duration-200 cursor-pointer hover:scale-105 active:scale-95 border"
                  style={{
                    borderColor: "var(--accent)",
                    color: "var(--accent-light)",
                  }}
                >
                  Try Again
                </button>
              </div>
            </div>
          )}
        </section>
      )}

      {/* ===== WHAT WE LEARNED ===== */}
      <section className="max-w-3xl mx-auto px-4 py-20">
        <h2 className="text-3xl md:text-4xl font-bold mb-8 text-center">
          What Did We Just Learn?
        </h2>

        <div className="grid md:grid-cols-2 gap-6 mb-10">
          {[
            { label: "Face ID", desc: "Unlocking your phone" },
            { label: "Social Media", desc: "Photo tagging on Facebook/Instagram" },
            { label: "Security", desc: "Airport & surveillance systems" },
            { label: "Filters", desc: "Snapchat & TikTok face filters" },
          ].map((item) => (
            <div
              key={item.label}
              className="rounded-xl p-4 border"
              style={{
                background: "var(--surface)",
                borderColor: "var(--border)",
              }}
            >
              <p className="font-bold">{item.label}</p>
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                {item.desc}
              </p>
            </div>
          ))}
        </div>

        <div
          className="rounded-2xl p-6 border"
          style={{
            background: "var(--surface)",
            borderColor: "var(--amber)",
            borderWidth: 1,
          }}
        >
          <p className="font-bold mb-2" style={{ color: "var(--amber)" }}>
            Privacy Matters
          </p>
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Your face embedding is a biometric fingerprint. Unlike a password,
            you can&rsquo;t change your face. All processing on this site happens{" "}
            <strong style={{ color: "var(--text)" }}>
              entirely in your browser
            </strong>{" "}
            — no images are ever uploaded to a server.
          </p>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer
        className="text-center py-8 border-t text-sm"
        style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
      >
        Built with face-api.js &middot; All processing happens in your browser
      </footer>
    </main>
  );
}
