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
/*  @vladmandic/human — ArcFace 512-dim + liveness (loaded from CDN)  */
/* ------------------------------------------------------------------ */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let human: any = null;

const HUMAN_CDN =
  "https://cdn.jsdelivr.net/npm/@vladmandic/human/dist/human.esm.js";
const MODEL_BASE =
  "https://cdn.jsdelivr.net/npm/@vladmandic/human/models/";
const LIVENESS_THRESHOLD = 0.5;

async function initHuman() {
  if (human) return human;

  // Dynamic ESM import from CDN — bypasses bundler entirely
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const H = await (Function('return import("' + HUMAN_CDN + '")')() as Promise<any>);
  const Human = H.Human || H.default;

  human = new Human({
    modelBasePath: MODEL_BASE,
    backend: "webgl",
    face: {
      enabled: true,
      detector: { enabled: true, maxDetected: 1, rotation: false },
      mesh: { enabled: true },
      description: { enabled: true }, // ArcFace 512-dim embeddings
      antispoof: { enabled: true },
      liveness: { enabled: true },
    },
    body: { enabled: false },
    hand: { enabled: false },
    gesture: { enabled: false },
    segmentation: { enabled: false },
  });

  await human.load();
  await human.warmup();
  return human;
}

async function getDescriptorAndLiveness(
  input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
): Promise<{
  descriptor: number[] | null;
  live: number;
  real: number;
}> {
  const result = await human.detect(input);
  if (!result.face || result.face.length === 0) {
    return { descriptor: null, live: 0, real: 0 };
  }
  const face = result.face[0];
  return {
    descriptor: face.embedding ?? null,
    live: face.live ?? 0,
    real: face.real ?? 0,
  };
}

function cosineSimilarity(a: number[], b: number[]): number {
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
  const [livenessScore, setLivenessScore] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const celebDescriptors = useRef<
    { name: string; descriptor: number[]; image: string }[]
  >([]);

  /* ---- Load models + celebrity embeddings ---- */
  const initialize = useCallback(async () => {
    setStage("loading-models");
    setError(null);
    try {
      setLoadProgress("Loading ArcFace recognition models...");
      await initHuman();

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

        const { descriptor } = await getDescriptorAndLiveness(img);
        if (descriptor) {
          descriptors.push({
            name: celeb.name,
            descriptor,
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
        "Camera access denied. Please allow camera access in your browser settings and try again."
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

  /* ---- Match ---- */
  const findMatch = async (canvas: HTMLCanvasElement) => {
    setStage("processing");
    setError(null);

    const { descriptor, live, real } = await getDescriptorAndLiveness(canvas);

    if (!descriptor) {
      setError(
        "No face detected. Make sure your face is clearly visible and well-lit, then try again."
      );
      setStage("ready");
      return;
    }

    // Liveness check
    const livenessAvg = (live + real) / 2;
    setLivenessScore(livenessAvg);

    if (livenessAvg < LIVENESS_THRESHOLD) {
      setError(
        "Liveness check failed — please use a real face in front of the camera, not a photo or screen."
      );
      setStage("ready");
      return;
    }

    const scored = celebDescriptors.current.map((c) => ({
      name: c.name,
      similarity: cosineSimilarity(descriptor, c.descriptor),
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
    setLivenessScore(null);
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
          style={{
            background: "var(--accent)",
            top: "-10%",
            left: "50%",
            transform: "translateX(-50%)",
          }}
        />

        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-4 animate-fade-in-up">
          Who&rsquo;s Your{" "}
          <span style={{ color: "var(--accent-light)" }}>Doppelganger</span>?
        </h1>
        <p
          className="text-lg md:text-xl max-w-2xl mb-8 animate-fade-in-up"
          style={{ color: "var(--text-muted)", animationDelay: "0.15s" }}
        >
          Discover which celebrity you look most like using ArcFace recognition
          with liveness detection — running entirely in your browser.
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
            <div
              className="w-8 h-8 border-3 border-t-transparent rounded-full animate-spin"
              style={{
                borderColor: "var(--accent)",
                borderTopColor: "transparent",
              }}
            />
            <p style={{ color: "var(--text-muted)" }}>{loadProgress}</p>
          </div>
        )}

        {error && <p className="mt-4 text-red-400 max-w-md">{error}</p>}

        {/* scroll hint */}
        {stage === "hero" && (
          <div
            className="absolute bottom-8 animate-bounce"
            style={{ color: "var(--text-muted)" }}
          >
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

        <div className="grid md:grid-cols-4 gap-6">
          {[
            {
              step: "1",
              title: "Detect",
              desc: "A neural network locates faces in the image, finding eyes, nose, and jawline.",
              icon: "👁️",
            },
            {
              step: "2",
              title: "Liveness",
              desc: "Anti-spoofing AI checks you're a real person — not a photo, screen, or mask.",
              icon: "🛡️",
            },
            {
              step: "3",
              title: "Embed",
              desc: "ArcFace converts your face into 512 numbers — a high-dimensional biometric fingerprint.",
              icon: "🧬",
            },
            {
              step: "4",
              title: "Compare",
              desc: "We measure how close two sets of 512 numbers are using cosine similarity.",
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
          <p
            className="font-mono text-sm mb-2"
            style={{ color: "var(--text-muted)" }}
          >
            Your face →
          </p>
          <p
            className="font-mono text-xs md:text-sm break-all"
            style={{ color: "var(--accent-light)" }}
          >
            [0.023, -0.145, 0.089, 0.234, -0.012, 0.178, ... 512 numbers
            total]
          </p>
          <p className="mt-3 text-sm" style={{ color: "var(--text-muted)" }}>
            <strong style={{ color: "var(--text)" }}>ArcFace</strong> produces
            4x more detail than older FaceNet models (512 vs 128 dimensions),
            powering the same class of tech behind Face ID and airport security.
          </p>
        </div>
      </section>

      {/* ===== CELEBRITY DATABASE ===== */}
      <section className="max-w-5xl mx-auto px-4 py-16">
        <h2 className="text-3xl md:text-4xl font-bold mb-3 text-center">
          The Comparison Database
        </h2>
        <p
          className="text-center mb-10"
          style={{ color: "var(--text-muted)" }}
        >
          15 public figures — ArcFace extracts 512 numbers from each face.
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
              <p className="text-xs md:text-sm font-medium truncate">
                {c.name}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ===== INTERACTIVE SECTION ===== */}
      {(stage === "ready" ||
        stage === "capturing" ||
        stage === "processing" ||
        stage === "results") && (
        <section id="try-it" className="max-w-3xl mx-auto px-4 py-20">
          <h2 className="text-3xl md:text-4xl font-bold mb-10 text-center">
            Find{" "}
            <span style={{ color: "var(--accent-light)" }}>Your</span>{" "}
            Doppelganger
          </h2>

          {/* --- READY: scan button --- */}
          {stage === "ready" && (
            <div className="flex flex-col items-center gap-4">
              <button
                onClick={startCamera}
                className="px-8 py-4 text-lg font-semibold rounded-xl transition-all duration-200 cursor-pointer hover:scale-105 active:scale-95"
                style={{ background: "var(--accent)", color: "#fff" }}
              >
                Scan Your Face
              </button>
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                Live camera scan with liveness detection — no uploads allowed
              </p>
              {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
            </div>
          )}

          {/* --- CAPTURING: video feed --- */}
          {stage === "capturing" && (
            <div className="flex flex-col items-center gap-4">
              <div
                className="rounded-2xl overflow-hidden border relative"
                style={{ borderColor: "var(--accent)", borderWidth: 2 }}
              >
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="mirror w-full max-w-md"
                />
                {/* scanning overlay */}
                <div className="absolute inset-0 pointer-events-none border-2 rounded-2xl" style={{ borderColor: "var(--green)", opacity: 0.4 }}>
                  <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2" style={{ borderColor: "var(--green)" }} />
                  <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2" style={{ borderColor: "var(--green)" }} />
                  <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2" style={{ borderColor: "var(--green)" }} />
                  <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2" style={{ borderColor: "var(--green)" }} />
                </div>
              </div>
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                Position your face within the frame
              </p>
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
                  Running liveness check &amp; ArcFace analysis...
                </p>
              </div>
            </div>
          )}

          {/* --- RESULTS --- */}
          {stage === "results" && results.length > 0 && (
            <div className="animate-fade-in-up">
              {/* Liveness badge */}
              {livenessScore !== null && (
                <div className="flex justify-center mb-6">
                  <span
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold"
                    style={{
                      background: "var(--surface)",
                      color: "var(--green)",
                      border: "1px solid var(--green)",
                    }}
                  >
                    <span>&#10003;</span> Liveness verified ({(livenessScore * 100).toFixed(0)}%)
                  </span>
                </div>
              )}

              {/* Top match hero */}
              <div
                className="rounded-2xl p-8 border mb-8 text-center"
                style={{
                  background: "var(--surface)",
                  borderColor: "var(--accent)",
                  borderWidth: 2,
                }}
              >
                <p
                  className="text-sm font-bold uppercase tracking-widest mb-4"
                  style={{ color: "var(--accent-light)" }}
                >
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
                  <div
                    className="text-4xl"
                    style={{ color: "var(--accent-light)" }}
                  >
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
                <div
                  className="px-6 py-4 border-b"
                  style={{ borderColor: "var(--border)" }}
                >
                  <h3 className="font-bold">Full Ranking</h3>
                </div>
                <div
                  className="divide-y"
                  style={{ borderColor: "var(--border)" }}
                >
                  {results.map((r, i) => (
                    <div
                      key={r.name}
                      className="flex items-center gap-4 px-6 py-3"
                      style={{ borderColor: "var(--border)" }}
                    >
                      <span
                        className="w-6 text-right text-sm font-bold"
                        style={{
                          color:
                            i < 3
                              ? "var(--accent-light)"
                              : "var(--text-muted)",
                        }}
                      >
                        {i + 1}
                      </span>
                      <img
                        src={r.image}
                        alt={r.name}
                        className="w-10 h-10 rounded-lg object-cover"
                      />
                      <span className="flex-1 font-medium text-sm">
                        {r.name}
                      </span>
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
                  Scan Again
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
            {
              label: "Social Media",
              desc: "Photo tagging on Facebook/Instagram",
            },
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
            you can&rsquo;t change your face. All processing on this site
            happens{" "}
            <strong style={{ color: "var(--text)" }}>
              entirely in your browser
            </strong>{" "}
            — no images are ever uploaded to a server. Liveness detection
            ensures only real, live faces are processed.
          </p>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer
        className="text-center py-8 border-t text-sm"
        style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
      >
        Powered by ArcFace (512-dim) &middot; Liveness detection enabled
        &middot; All processing happens in your browser
      </footer>
    </main>
  );
}
