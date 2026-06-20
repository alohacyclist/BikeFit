/**
 * VideoRecorder — kapselt MediaRecorder zur Aufnahme der Webcam in eine Datei,
 * die später für die Replay-Pipeline verwendet wird.
 *
 * WICHTIG: Browser-MediaRecorder produziert i.d.R. VFR (variable frame rate).
 * Für frame-deterministisches Replay muss die Ausgabe anschließend extern
 * mit ffmpeg auf CFR konvertiert werden:
 *   ffmpeg -i input.webm -c:v libx264 -r 30 -vsync cfr -g 1 -pix_fmt yuv420p replay.mp4
 *
 * Diese Pflicht ist in der Studienprotokoll-Dokumentation festzuhalten.
 */

export interface VideoRecorderOptions {
  mimeType?: string;
  videoBitsPerSecond?: number;
}

const DEFAULT_MIME = 'video/webm;codecs=vp9';
const FALLBACK_MIME = 'video/webm;codecs=vp8';

function pickSupportedMime(preferred?: string): string {
  const candidates = [preferred, DEFAULT_MIME, FALLBACK_MIME, 'video/webm'].filter(
    (m): m is string => typeof m === 'string'
  );
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported(mime)) return mime;
  }
  throw new Error('Browser unterstützt kein MediaRecorder-Format');
}

export class VideoRecorder {
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private mimeType: string = DEFAULT_MIME;
  private startedAt: number = 0;

  start(stream: MediaStream, options: VideoRecorderOptions = {}): void {
    if (this.recorder) {
      throw new Error('Aufnahme läuft bereits');
    }
    this.mimeType = pickSupportedMime(options.mimeType);
    this.chunks = [];
    this.recorder = new MediaRecorder(stream, {
      mimeType: this.mimeType,
      videoBitsPerSecond: options.videoBitsPerSecond ?? 5_000_000,
    });
    this.recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) this.chunks.push(e.data);
    };
    this.startedAt = performance.now();
    this.recorder.start();
  }

  isRecording(): boolean {
    return this.recorder !== null && this.recorder.state === 'recording';
  }

  elapsedMs(): number {
    return this.startedAt === 0 ? 0 : performance.now() - this.startedAt;
  }

  async stop(): Promise<Blob> {
    if (!this.recorder) {
      throw new Error('Keine aktive Aufnahme');
    }
    const recorder = this.recorder;
    const blob = await new Promise<Blob>((resolve) => {
      recorder.onstop = () => {
        resolve(new Blob(this.chunks, { type: this.mimeType }));
      };
      recorder.stop();
    });
    this.recorder = null;
    this.chunks = [];
    this.startedAt = 0;
    return blob;
  }

  /** Triggert Browser-Download des fertigen Blobs. */
  downloadBlob(blob: Blob, participantId: string): void {
    const url = URL.createObjectURL(blob);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const ext = this.mimeType.includes('webm') ? 'webm' : 'mp4';
    const a = document.createElement('a');
    a.href = url;
    a.download = `replay_${participantId}_${timestamp}.${ext}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}
