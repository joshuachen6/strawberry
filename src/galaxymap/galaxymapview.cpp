#include "galaxymapview.h"

#include <cmath>
#include <algorithm>

#include <QApplication>
#include <QPainter>
#include <QPainterPath>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QShowEvent>
#include <QRandomGenerator>
#include <QPixmap>
#include <QImage>
#include <QImageReader>
#include <QRadialGradient>
#include <QLinearGradient>
#include <QConicalGradient>
#include <QFontMetrics>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <QColor>
#include <QSet>
#include <QDebug>
#include <QSettings>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QVariant>
#include <QProcess>
#include <QStandardPaths>

#include "constants/appearancesettings.h"
#include "core/settings.h"

#include "core/application.h"
#include "collection/collectionmodel.h"
#include "collection/collectionbackend.h"
#include "collection/collectionitem.h"
#include "core/song.h"
#include "core/logging.h"
#include "playlist/playlistmanager.h"
#include "covermanager/albumcoverloader.h"
#include "covermanager/albumcoverloaderoptions.h"
#include "covermanager/albumcoverloaderresult.h"
#include <QFile>
#ifdef HAVE_MOODBAR
#include "moodbar/moodbarloader.h"
#endif

#ifdef HAVE_AUBIO
#include <aubio/aubio.h>

struct AubioBPMResult { float bpm; float confidence; };

static AubioBPMResult getAubioBPMData(const QUrl& url) {
    if (!url.isLocalFile()) return {120.0f, 0.0f};
    QString path = url.toLocalFile();

    static QHash<QString, AubioBPMResult> s_bpmCache;
    if (s_bpmCache.contains(path)) return s_bpmCache.value(path);

    uint_t samplerate = 0, hop_size = 256;
    QByteArray pathBa = path.toUtf8();
    aubio_source_t *source = new_aubio_source(pathBa.constData(), samplerate, hop_size);
    if (!source) { s_bpmCache.insert(path, {120.0f, 0.0f}); return {120.0f, 0.0f}; }

    samplerate = aubio_source_get_samplerate(source);
    if (samplerate == 0) samplerate = 44100;

    aubio_tempo_t *tempo = new_aubio_tempo("default", 1024, hop_size, samplerate);
    if (!tempo) { del_aubio_source(source); s_bpmCache.insert(path, {120.0f, 0.0f}); return {120.0f, 0.0f}; }

    fvec_t *in  = new_fvec(hop_size);
    fvec_t *out = new_fvec(2);

    const float  kMinConfidence = 0.3f;
    const uint_t kBatchBlocks   = (10 * samplerate) / hop_size;
    const uint_t kMaxBatches    = 12;
    uint_t start_samples = samplerate * 20;

    float bpm = 0.0f, confidence = 0.0f;
    for (uint_t batch = 0; batch < kMaxBatches; ++batch) {
        aubio_source_seek(source, start_samples + batch * kBatchBlocks * hop_size);
        uint_t read = 0;
        for (uint_t b = 0; b < kBatchBlocks; ++b) {
            aubio_source_do(source, in, &read);
            aubio_tempo_do(tempo, in, out);
            if (read < hop_size) goto done;
        }
        bpm        = aubio_tempo_get_bpm(tempo);
        confidence = aubio_tempo_get_confidence(tempo);
        if (confidence >= kMinConfidence && bpm > 0.0f) break;
    }
done:
    bpm        = aubio_tempo_get_bpm(tempo);
    confidence = aubio_tempo_get_confidence(tempo);

    del_aubio_tempo(tempo);
    del_aubio_source(source);
    del_fvec(in);
    del_fvec(out);

    if (bpm <= 0.0f || std::isinf(bpm) || std::isnan(bpm)) { bpm = 120.0f; confidence = 0.0f; }
    AubioBPMResult res{bpm, confidence};
    s_bpmCache.insert(path, res);
    return res;
}

[[maybe_unused]] static float getAubioBPM(const QUrl& url) { return getAubioBPMData(url).bpm; }
#else
struct AubioBPMResult { float bpm; float confidence; };
static AubioBPMResult getAubioBPMData(const QUrl&) { return {120.0f, 0.0f}; }
static float getAubioBPM(const QUrl&) { return 120.0f; }
#endif

using namespace Qt::Literals::StringLiterals;
#include <QImageReader>

namespace {

// Extract dominant colour from a QPixmap (fast approximate)
QVector3D dominantColor(const QPixmap &pix) {
  if (pix.isNull()) return QVector3D(0.5f, 0.5f, 1.0f);
  QImage img = pix.scaled(8, 8, Qt::IgnoreAspectRatio, Qt::FastTransformation).toImage();
  long long r = 0, g = 0, b = 0, n = 0;
  for (int y = 0; y < img.height(); ++y) {
    for (int x = 0; x < img.width(); ++x) {
      QColor c = img.pixelColor(x, y);
      r += c.red(); g += c.green(); b += c.blue(); ++n;
    }
  }
  if (n == 0) return QVector3D(0.5f, 0.7f, 1.0f);
  return QVector3D(r / (255.0f * n), g / (255.0f * n), b / (255.0f * n));
}

}  // namespace

GalaxyMapView::GalaxyMapView(Application *app, QWidget *parent)
    : QWidget(parent),
      app_(app),
      timer_(new QTimer(this)),
      zoom_(0.06f),
      pan_(0.0f, 0.0f),
      velocity_(0.0f, 0.0f),
      is_dragging_(false),
      anim_time_(0.0f),
      hovered_star_(-1),
      selected_star_(-1),
      select_pulse_(0.0f) {
  setMouseTracking(true);
  setAttribute(Qt::WA_OpaquePaintEvent);
  setMinimumSize(200, 200);

  QObject::connect(timer_, &QTimer::timeout, this, &GalaxyMapView::updateFrame);
  timer_->start(16);  // ~60fps
}

GalaxyMapView::~GalaxyMapView() = default;

void GalaxyMapView::Init() {
  CollectionBackend *backend = app_->collection_backend().get();
  CollectionModel  *model   = app_->collection_model();

  // Real-time updates: model emits SongsAdded/Removed as library changes
  QObject::connect(model,   &CollectionModel::SongsAdded,   this, &GalaxyMapView::onSongsChanged);
  QObject::connect(model,   &CollectionModel::SongsRemoved, this, &GalaxyMapView::onSongsChanged);
  // Backend SongsAdded fires as new songs are scanned into the library
  QObject::connect(backend, &CollectionBackend::SongsAdded, this, &GalaxyMapView::onSongsChanged);
  QObject::connect(backend, &CollectionBackend::SongsDeleted, this, &GalaxyMapView::onSongsChanged);
  QObject::connect(backend, &CollectionBackend::SongsChanged, this, &GalaxyMapView::onSongsChanged);
  QObject::connect(backend, &CollectionBackend::DatabaseReset, this, [this]() {
    fetchSongsFromBackend();
  });

  // Connect album cover loader
  QObject::connect(app_->albumcover_loader().get(), &AlbumCoverLoader::AlbumCoverLoaded,
                   this, &GalaxyMapView::onAlbumCoverLoaded);

  // Step 1: Show placeholder dots immediately so the canvas isn't blank
  buildPlaceholderStars();

  // Step 2: Try loading from already-populated model nodes
  tryLoadFromModel();

  // Step 3: Kick off async backend fetch for full song data
  fetchSongsFromBackend();
}

void GalaxyMapView::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  if (all_songs_.isEmpty()) {
    tryLoadFromModel();
    fetchSongsFromBackend();
  }
}

void GalaxyMapView::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  update();
}

void GalaxyMapView::leaveEvent(QEvent *event) {
  QWidget::leaveEvent(event);
  hovered_star_ = -1;
  update();
}


// ─── Data Loading ─────────────────────────────────────────────────────────────

void GalaxyMapView::buildPlaceholderStars() {
  stars_.clear();
  update();
}

void GalaxyMapView::tryLoadFromModel() {
  CollectionModel *model = app_->collection_model();
  const QList<CollectionItem*> nodes = model->song_nodes();
  qLog(Debug) << "GalaxyMapView::tryLoadFromModel:" << nodes.size() << "nodes";
  if (nodes.isEmpty()) return;

  SongList songs;
  songs.reserve(nodes.size());
  for (CollectionItem *item : nodes) {
    if (item) songs.append(item->metadata);
  }
  if (!songs.isEmpty()) {
    qLog(Debug) << "GalaxyMapView: building stars from" << songs.size() << "model songs";
    all_songs_ = songs;
    buildStars(songs);
  }
}

void GalaxyMapView::onSongsChanged(const SongList &changed) {
  Q_UNUSED(changed)
  fetchSongsFromBackend();
}

void GalaxyMapView::fetchSongsFromBackend() {
  CollectionBackend *backend = app_->collection_backend().get();
  // Invoke GetAllSongs on the backend's own thread via queued connection
  QMetaObject::invokeMethod(backend, [this, backend]() {
    SongList songs = backend->GetAllSongs();
    qLog(Debug) << "GalaxyMapView::fetchSongsFromBackend: got" << songs.size() << "songs";
    // Deliver results to the main thread
    QMetaObject::invokeMethod(this, [this, songs]() {
      all_songs_ = songs;
      buildStars(songs);
    }, Qt::QueuedConnection);
  }, Qt::QueuedConnection);
}

void GalaxyMapView::buildStars(const SongList &songs) {
  stars_.clear();
  album_art_cache_.clear();
  album_art_thumb_cache_.clear();
  album_art_loading_.clear();
  loader_id_to_album_key_.clear();
  update();

  // Version sentinel — clear stale cache if math changed
  const int kVibeVersion = 8;
  {
    QSettings vc(u"Strawberry"_s, u"GalaxyVibeMap"_s);
    if (vc.value(u"__version"_s).toInt() != kVibeVersion) {
      vc.clear();
      vc.setValue(u"__version"_s, kVibeVersion);
    }
  }

  // ── PHASE 0 (main thread): read MoodbarLoader and QSettings ─────────────────
  // MoodbarLoader is a QObject — must stay on main thread.
  // We collect plain byte arrays + booleans; no Qt objects leave this scope.
  struct RawEntry {
    QByteArray mood_data;
    bool       from_cache;
    float      cached_x, cached_y;
    float      cached_r, cached_g, cached_b;
    bool       will_async;
    MoodbarPipelinePtr pipeline;
  };

  QVector<RawEntry> raw;
  raw.reserve(songs.size());

  QVector<uint> jitter_seeds;
  jitter_seeds.reserve(songs.size());

  {
    QSettings vibe_cache(u"Strawberry"_s, u"GalaxyVibeMap"_s);

    for (int i = 0; i < songs.size(); ++i) {
      const Song &song = songs[i];
      jitter_seeds.append(qHash(song.url().toEncoded()));

      RawEntry e{};
      if (!song.is_valid() || song.unavailable()) { raw.append(e); continue; }

      QString hk = u"vcode_"_s + QString::fromUtf8(song.url().toEncoded().toHex());
      if (vibe_cache.value(hk + u"_acoustic"_s).toBool()) {
        e.from_cache = true;
        e.cached_x = vibe_cache.value(hk + u"_X"_s).toFloat();
        e.cached_y = vibe_cache.value(hk + u"_Y"_s).toFloat();
        e.cached_r = vibe_cache.value(hk + u"_R"_s).toFloat();
        e.cached_g = vibe_cache.value(hk + u"_G"_s).toFloat();
        e.cached_b = vibe_cache.value(hk + u"_B"_s).toFloat();
        raw.append(e); continue;
      }

#ifdef HAVE_MOODBAR
      if (app_->moodbar_loader()) {
        MoodbarLoader::LoadResult res = app_->moodbar_loader()->Load(song.url(), song.has_cue());
        if (res.status == MoodbarLoader::LoadStatus::Loaded && !res.data.isEmpty()) {
          e.mood_data = res.data;
        } else if (res.status == MoodbarLoader::LoadStatus::WillLoadAsync && res.pipeline) {
          e.will_async = true;
          e.pipeline = res.pipeline;
        }
      }
#endif
      raw.append(e);
    }
  }

  // Reusable math logic for both blocking load and async load
  auto processMoodMath = [](const QByteArray &data, float &x, float &mood_y, QVector3D &col) {
    int n = data.size() / 3;
    if (n <= 0) return false;

    double rSum = 0, gSum = 0, bSum = 0;
    double bSumSq = 0;

    for (int j = 0; j < n; ++j) {
      double r = static_cast<unsigned char>(data[j*3]);
      double g = static_cast<unsigned char>(data[j*3+1]);
      double b = static_cast<unsigned char>(data[j*3+2]);
      rSum += r; gSum += g; bSum += b;
      bSumSq += b * b;
    }

    double meanR = rSum / n, meanG = gSum / n, meanB = bSum / n;
    double normR = meanR / 255.0, normG = meanG / 255.0, normB = meanB / 255.0;

    // X-axis: Spectral Shape (Spectral Skewness)
    x = static_cast<float>((normB - normR) * (1.0 + (normB / (normG + 0.1))) * 500.0);

    // Y-axis: Pulse Entropy (Flux Variance relative to Mean Flux)
    double fluxSum = 0.0, fluxSumSq = 0.0;
    double prevLum = (static_cast<unsigned char>(data[0])*0.2126
                    + static_cast<unsigned char>(data[1])*0.7152
                    + static_cast<unsigned char>(data[2])*0.0722);
    for (int j = 1; j < n; ++j) {
      double lum = (static_cast<unsigned char>(data[j*3  ])*0.2126
                  + static_cast<unsigned char>(data[j*3+1])*0.7152
                  + static_cast<unsigned char>(data[j*3+2])*0.0722);
      double flux = std::abs(lum - prevLum);
      fluxSum += flux;
      fluxSumSq += flux * flux;
      prevLum = lum;
    }
    double avgFlux = fluxSum / (n - 1);
    double fluxVar = std::max(0.0, (fluxSumSq / (n - 1)) - (avgFlux * avgFlux));
    mood_y = static_cast<float>(1000.0 * std::log10(1.0 + (fluxVar / (avgFlux + 1.0))));

    // Frequency Jitter: B-channel variance offset
    double bVar = std::max(0.0, (bSumSq / n) - (meanB * meanB));
    float bJitter = static_cast<float>(std::sqrt(bVar) / 255.0 * 60.0 - 30.0);
    x += bJitter;

    if (std::isnan(x) || std::isinf(x)) x = 0;
    if (std::isnan(mood_y) || std::isinf(mood_y)) mood_y = 0;

    // Z-axis/Color: Spectral Contrast (Saturation)
    double meanColor = (normR + normG + normB) / 3.0;
    double varColor = ((normR - meanColor)*(normR - meanColor) 
                     + (normG - meanColor)*(normG - meanColor) 
                     + (normB - meanColor)*(normB - meanColor)) / 3.0;
    double stdColor = std::sqrt(varColor);
    double saturation = std::clamp(stdColor * 2.5, 0.0, 1.0);

    QColor tc; tc.setRgbF(normR, normG, normB);
    if (!tc.isValid()) tc = QColor(100, 150, 255);
    float h, s, v; tc.getHsvF(&h, &s, &v);
    s = static_cast<float>(saturation);
    tc.setHsvF(h, s, v);
    col = QVector3D(tc.redF(), tc.greenF(), tc.blueF());

    return true;
  };

  // Set up async pipeline callbacks on main thread (they update stars_ later)
  for (int i = 0; i < raw.size(); ++i) {
    if (!raw[i].will_async || !raw[i].pipeline) continue;
    const Song &song = songs[i];
    MoodbarPipelinePtr pp = raw[i].pipeline;
    QString hk = u"vcode_"_s + QString::fromUtf8(song.url().toEncoded().toHex());
    connect(pp.get(), &MoodbarPipeline::Finished, this, [this, i, hk, pp, processMoodMath](bool success) {
      if (!success || i >= stars_.size()) return;
      const QByteArray &data = pp->data();
      
      float ax = 0, ay = 0;
      QVector3D ac;
      if (processMoodMath(data, ax, ay, ac)) {
        QSettings vc(u"Strawberry"_s, u"GalaxyVibeMap"_s);
        vc.setValue(hk+u"_X"_s, ax); vc.setValue(hk+u"_Y"_s, ay);
        vc.setValue(hk+u"_R"_s, ac.x()); vc.setValue(hk+u"_G"_s, ac.y()); vc.setValue(hk+u"_B"_s, ac.z());
        vc.setValue(hk+u"_acoustic"_s, true);
        stars_[i].position = QVector2D(ax, ay);
        stars_[i].color = ac;
        update();
      }
    }); // Qt::AutoConnection is fine here since both emit and catch are on main thread
  }

  // ── PHASE 1 (background): coordinate math ──────────────────────────────
  Settings s;
  s.beginGroup(AppearanceSettings::kSettingsGroup);
  AppearanceSettings::GalaxyBackendType backend_type = static_cast<AppearanceSettings::GalaxyBackendType>(s.value(AppearanceSettings::kGalaxyBackend, static_cast<int>(AppearanceSettings::GalaxyBackendType::BasicMath)).toInt());
  s.endGroup();

  if (backend_type == AppearanceSettings::GalaxyBackendType::DeepEmbeddings) {
      static bool deep_embedding_script_started = false;
      if (!deep_embedding_script_started) {
          deep_embedding_script_started = true;
          QString music_dir = QStandardPaths::writableLocation(QStandardPaths::MusicLocation);
          QString script_path = QStringLiteral("/home/jchen/Documents/Projects/strawberry/src/galaxymap/analyze_library.py");
          QProcess::startDetached(QStringLiteral("python3"), {script_path, QStringLiteral("--dir"), music_dir});
      }
  }

  auto future_ = QtConcurrent::run([this, songs, raw, jitter_seeds, processMoodMath, backend_type]() {
    struct SongEntry { QVector2D pos; QVector3D col; float bpm; };
    QVector<SongEntry> entries;
    entries.reserve(songs.size());
    QRandomGenerator bg_rng(0xDEADBEEF);
    QSettings vibe_cache(u"Strawberry"_s, u"GalaxyVibeMap"_s);

    QHash<QString, QPair<QVector2D, QVector3D>> db_embeddings;
    if (backend_type == AppearanceSettings::GalaxyBackendType::DeepEmbeddings) {
      QString db_path = QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + QStringLiteral("/strawberry/strawberry/galaxy_embeddings.db");
      if (QFile::exists(db_path)) {
        {
          QSqlDatabase db = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), QStringLiteral("embeddings_db"));
          db.setDatabaseName(db_path);
          if (db.open()) {
            QSqlQuery q(db);
            q.exec(QStringLiteral("SELECT path, x, y, vibrancy FROM embeddings WHERE x IS NOT NULL"));
            while (q.next()) {
              QString path = q.value(0).toString();
              float x = q.value(1).toFloat();
              float y = q.value(2).toFloat();
              float vib = q.value(3).toFloat();
              
              float pb = std::clamp(vib / 2.0f, 0.0f, 1.0f);
              QVector3D col((1-pb)*0.8f+0.2f, 0.4f, pb*0.8f+0.2f);
              db_embeddings[path] = {QVector2D(x, y), col};
            }
          }
        }
        QSqlDatabase::removeDatabase(QStringLiteral("embeddings_db"));
      }
    }

    for (int i = 0; i < songs.size(); ++i) {
      const Song &song = songs[i];
      const RawEntry &re = raw[i];

      if (!song.is_valid() || song.unavailable()) {
        entries.append({QVector2D(0,0), QVector3D(0.5f,0.5f,0.5f), 120.0f});
        continue;
      }

      float bpm = song.bpm() > 0.0f ? song.bpm() : 120.0f;
      QString hk = u"vcode_"_s + QString::fromUtf8(song.url().toEncoded().toHex());
      
      if (backend_type == AppearanceSettings::GalaxyBackendType::DeepEmbeddings) {
        QString local_path = song.url().toLocalFile();
        if (db_embeddings.contains(local_path)) {
          QPair<QVector2D, QVector3D> emb = db_embeddings.value(local_path);
          entries.append({emb.first, emb.second, bpm});
          continue;
        } else {
          // Fallback if not processed yet or not local
          float x = (static_cast<float>(bg_rng.generateDouble()) - 0.5f) * 1500.0f;
          float y = (static_cast<float>(bg_rng.generateDouble()) - 0.5f) * 1500.0f;
          entries.append({QVector2D(x, y), QVector3D(0.5f,0.5f,0.5f), bpm});
          continue;
        }
      }

      if (re.from_cache) {
        entries.append({QVector2D(re.cached_x, re.cached_y), QVector3D(re.cached_r, re.cached_g, re.cached_b), bpm});
        continue;
      }

      float x = 0, mood_y = 0;
      QVector3D col(0.5f,0.5f,0.5f);
      bool has_acoustics = false;

      if (!re.mood_data.isEmpty()) {
        if (processMoodMath(re.mood_data, x, mood_y, col)) {
          has_acoustics = true;
          vibe_cache.setValue(hk+u"_X"_s, x);
          vibe_cache.setValue(hk+u"_Y"_s, mood_y);
          vibe_cache.setValue(hk+u"_R"_s, col.x());
          vibe_cache.setValue(hk+u"_G"_s, col.y());
          vibe_cache.setValue(hk+u"_B"_s, col.z());
          vibe_cache.setValue(hk+u"_acoustic"_s, true);
        }
      }

      if (!has_acoustics) {
        x      = (static_cast<float>(bg_rng.generateDouble()) - 0.5f) * 1500.0f;
        mood_y = (static_cast<float>(bg_rng.generateDouble()) - 0.5f) * 1500.0f;
        float pb = std::clamp((bpm-60.0f)/120.0f, 0.0f, 1.0f);
        col = QVector3D((1-pb)*0.8f+0.2f, 0.4f, pb*0.8f+0.2f);
      }
      entries.append({QVector2D(x, mood_y), col, bpm});
    }

    // ── PHASE 2 (main thread): build star objects immediately ─────────────────
    QMetaObject::invokeMethod(this, [this, songs, entries, jitter_seeds]() {
      QRandomGenerator rng(0xDEADBEEF);

      QVector<float> all_x, all_y;
      all_x.reserve(songs.size()); all_y.reserve(songs.size());
      for (int i = 0; i < songs.size(); ++i) {
        if (!songs[i].is_valid() || songs[i].unavailable()) continue;
        all_x.append(entries[i].pos.x());
        all_y.append(entries[i].pos.y());
      }
      float med_x=0, med_y=0;
      if (!all_x.isEmpty()) {
        std::sort(all_x.begin(), all_x.end());
        std::sort(all_y.begin(), all_y.end());
        med_x = all_x[all_x.size()/2];
        med_y = all_y[all_y.size()/2];
      }

      for (int i = 0; i < songs.size(); ++i) {
        const Song &song = songs[i];
        if (!song.is_valid() || song.unavailable()) continue;
        float bpm = entries[i].bpm;
        if (bpm <= 0) bpm = 120.0f;
        float bx = entries[i].pos.x() - med_x;
        float by = entries[i].pos.y() - med_y;
        uint hs = jitter_seeds[i];
        auto jitter = [&](uint seed) {
          return (static_cast<float>(qHash(hs^seed)%10000)/10000.0f-0.5f)*40.0f;
        };
        float x = bx + jitter(0x1234);
        float y = by + jitter(0x5678) + (bpm-120.0f)*2.0f;

        QColor col;
        col.setRgbF(entries[i].col.x(), entries[i].col.y(), entries[i].col.z());
        if (!col.isValid()) col = QColor(100,150,255);

        GalaxyStar star;
        star.position      = QVector2D(x,y);
        star.color         = QVector3D(col.redF(), col.greenF(), col.blueF());
        star.base_size     = 3.0f + std::min(1.0f, std::max(0.0f, song.rating()))*2.0f;
        star.twinkle_phase = static_cast<float>(rng.generateDouble()*2.0*M_PI);
        star.title         = song.title();
        star.artist        = song.artist();
        star.album         = song.album();
        star.album_artist  = song.effective_albumartist();
        star.bpm           = bpm;
        star.song_id       = song.id();
        star.all_songs_index = i;
        star.album_key     = song.AlbumKey();
        if (star.album_key.isEmpty()) star.album_key = song.artist() + u"|" + song.album();
        if (!song.art_manual().isEmpty())         star.art_url = song.art_manual();
        else if (!song.art_automatic().isEmpty()) star.art_url = song.art_automatic();
        star.url = song.url();
        stars_.append(star);
      }

      constellations_.clear();
      constellations_.append({{},{},QVector2D(-1000,-1000),u"Distorted / Gritty"_s});
      constellations_.append({{},{},QVector2D(-1000, 1000),u"Acoustic / Warm"_s});
      constellations_.append({{},{},QVector2D( 1000,-1000),u"Bright / Sharp"_s});
      constellations_.append({{},{},QVector2D( 1000, 1000),u"Tonal / Pure"_s});
      update();

      // ── PHASE 3a: Vibe Gravitation — pull co-located stars 5% toward cluster centroid
      {
        QHash<QPair<int,int>, QVector<int>> clusters; // rounded pos → star indices
        for (int i = 0; i < stars_.size(); ++i) {
          int rx = static_cast<int>(std::round(stars_[i].position.x() * 10.0f));
          int ry = static_cast<int>(std::round(stars_[i].position.y() * 10.0f));
          clusters[{rx, ry}].append(i);
        }
        for (auto it = clusters.cbegin(); it != clusters.cend(); ++it) {
          const QVector<int> &idx = it.value();
          if (idx.size() < 2) continue;
          QVector2D centroid(0,0);
          for (int i : idx) centroid += stars_[i].position;
          centroid /= static_cast<float>(idx.size());
          for (int i : idx)
            stars_[i].position = stars_[i].position * 0.95f + centroid * 0.05f;
        }
      }

      // ── PHASE 3b: refine BPM per-star via Aubio (non-blocking, confidence-weighted)
      for (int i = 0; i < stars_.size(); ++i) {
        const QUrl url = stars_[i].url;
        if (!url.isLocalFile()) continue;
        QtConcurrent::run([this, i, url]() {
          AubioBPMResult res = getAubioBPMData(url);
          if (res.bpm > 0.0f) {
            QMetaObject::invokeMethod(this, [this, i, res]() {
              if (i >= stars_.size()) return;
              stars_[i].bpm = res.bpm;
              // Confidence-weighted Y nudge: if the beat is confident and strong,
              // pull the star slightly toward the BPM-derived position.
              // Beatless/ambient tracks (low confidence) are left where spectral data put them.
              // Trust spectral flux variance unless the beat is extremely confident (>0.7)
              if (res.confidence > 0.7f) {
                float bpm_y = (res.bpm - 120.0f) * 4.0f;
                float blend = std::min(res.confidence * 0.7f, 0.5f); // up to 50%
                stars_[i].position = QVector2D(
                  stars_[i].position.x(),
                  stars_[i].position.y() * (1.0f - blend) + bpm_y * blend);
              }
              update();
            }, Qt::QueuedConnection);
          }
        });
      }
    }, Qt::QueuedConnection);
  });
  Q_UNUSED(future_)
}

// ─── Art Loading ──────────────────────────────────────────────────────────────

void GalaxyMapView::loadStarArt(int idx) {
  if (idx < 0 || idx >= stars_.size()) return;
  
  const GalaxyStar &star = stars_[idx];
  const QString &album_key = star.album_key;

  if (album_art_cache_.contains(album_key)) {
    return;
  }
  
  if (album_art_loading_.contains(album_key)) return;

  if (star.all_songs_index < 0 || star.all_songs_index >= all_songs_.size()) return;
  const Song &song = all_songs_[star.all_songs_index];

  album_art_loading_.insert(album_key);

  AlbumCoverLoaderOptions options;
  options.desired_scaled_size = QSize(256, 256);
  options.options |= AlbumCoverLoaderOptions::Option::ScaledImage;

  quint64 id = app_->albumcover_loader()->LoadImageAsync(options, song);
  loader_id_to_album_key_[id] = album_key;
}

void GalaxyMapView::onAlbumCoverLoaded(quint64 id, const AlbumCoverLoaderResult &result) {
  if (!loader_id_to_album_key_.contains(id)) return;

  QString album_key = loader_id_to_album_key_.take(id);
  album_art_loading_.remove(album_key);
  
  QPixmap pix;
  if (result.success && !result.image_scaled.isNull()) {
    pix = QPixmap::fromImage(result.image_scaled);
  }
  
  if (pix.isNull()) {
    // Try to find ANY song from the original songs that has a valid art_url
    for (const Song &s : all_songs_) {
      if (s.AlbumKey() == album_key || (s.artist() + u"|" + s.album()) == album_key) {
        QString path = s.art_manual().toLocalFile();
        if (path.isEmpty()) path = s.art_automatic().toLocalFile();
        if (!path.isEmpty() && QFile::exists(path)) {
           QImageReader reader(path);
           reader.setAutoTransform(true);
           reader.setScaledSize(QSize(256, 256));
           QImage img = reader.read();
           if (!img.isNull()) {
             pix = QPixmap::fromImage(img);
             break;
           }
        }
      }
    }
  }

  if (!pix.isNull()) {
    onAlbumArtLoaded(album_key, pix);
  }
}

void GalaxyMapView::onAlbumArtLoaded(const QString &album_key, const QPixmap &pixmap) {
  if (pixmap.isNull()) return;

  album_art_cache_[album_key] = pixmap;
  album_art_thumb_cache_[album_key] = pixmap.scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation);

  // Update stars count and dominant colors
  QVector3D dc = dominantColor(pixmap);
  for (int i = 0; i < stars_.size(); ++i) {
    if (stars_[i].album_key == album_key) {
      stars_[i].color = dc;
    }
  }
  update();
}

// ─── Coordinate Helpers ───────────────────────────────────────────────────────

QPointF GalaxyMapView::worldToScreen(QVector2D world) const {
  QPointF center(width() / 2.0, height() / 2.0);
  return center + QPointF((world.x() - pan_.x()) * zoom_,
                          -(world.y() - pan_.y()) * zoom_);  // Y-flip
}

QVector2D GalaxyMapView::screenToWorld(QPointF screen) const {
  QPointF center(width() / 2.0, height() / 2.0);
  return QVector2D(
    static_cast<float>((screen.x() - center.x()) / zoom_) + pan_.x(),
    static_cast<float>(-(screen.y() - center.y()) / zoom_) + pan_.y()
  );
}

// ─── Update Loop ──────────────────────────────────────────────────────────────

void GalaxyMapView::updateFrame() {
  anim_time_ += 0.016f;

  if (!is_dragging_) {
    pan_ += velocity_;
    velocity_ *= 0.92f;
  }

  if (selected_star_ >= 0) {
    select_pulse_ = fmod(select_pulse_ + 0.025f, 1.0f);
  }

  update();
}

// ─── Painting ─────────────────────────────────────────────────────────────────

void GalaxyMapView::paintEvent(QPaintEvent *) {
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  p.setRenderHint(QPainter::SmoothPixmapTransform, true);

  // Deep space background
  QLinearGradient bg(0, 0, width(), height());
  bg.setColorAt(0.0, QColor(2, 3, 15));
  bg.setColorAt(0.5, QColor(5, 6, 25));
  bg.setColorAt(1.0, QColor(3, 5, 20));
  p.fillRect(rect(), bg);

  drawParallaxBackground(p);

  if (zoom_ < kZoomStarView) {
    drawStarView(p);
  } else if (zoom_ < kZoomConstellation) {
    drawStarView(p);
    drawConstellationView(p);
  } else {
    drawPlanetView(p);
  }

  // HUD: view mode and star count
  QString mode = zoom_ < kZoomStarView ? u"\u2726 Star Field"_s
               : zoom_ < kZoomConstellation ? u"\u2728 Constellation View"_s
               : u"\u25c9 Planet View"_s;
  QString hud = mode + u"  |  "_s + QString::number(stars_.size()) + u" stars"_s;
  p.setPen(QColor(255, 255, 255, 220));
  p.setFont(QFont(u"Inter"_s, 9));
  p.drawText(rect().adjusted(8, 8, -8, -8), Qt::AlignTop | Qt::AlignRight, hud);
}




// ─── Parallax Background ──────────────────────────────────────────────────────

void GalaxyMapView::drawParallaxBackground(QPainter &p) {
  p.save();
  
  auto drawLayer = [&](float speed, float tileSize, int count, bool is_nebula) {
    float offsetX = -pan_.x() * zoom_ * speed;
    float offsetY = pan_.y() * zoom_ * speed;
    
    int startGridX = qFloor(-offsetX / tileSize);
    int endGridX   = qFloor((width() - offsetX) / tileSize);
    int startGridY = qFloor(-offsetY / tileSize);
    int endGridY   = qFloor((height() - offsetY) / tileSize);

    for (int tx = startGridX; tx <= endGridX; ++tx) {
      for (int ty = startGridY; ty <= endGridY; ++ty) {
        uint seed = qHash(static_cast<qint64>(tx) << 32 | static_cast<quint32>(ty));
        if (is_nebula) seed ^= 0xFE98CD;
        else seed ^= 0xAB12CD;
        
        QRandomGenerator rng(seed);
        for (int i = 0; i < count; ++i) {
          float lx = static_cast<float>(rng.generateDouble()) * tileSize;
          float ly = static_cast<float>(rng.generateDouble()) * tileSize;
          float sx = tx * tileSize + lx + offsetX;
          float sy = ty * tileSize + ly + offsetY;
          
          if (is_nebula) {
            float hue = std::clamp(static_cast<float>(rng.generateDouble()), 0.0f, 0.999f);
            int nr = 80 + static_cast<int>(rng.generateDouble() * 200);
            QRadialGradient nebula(QPointF(sx, sy), nr * 2.5f);
            QColor nc = QColor::fromHsvF(hue, 0.7f, 0.6f, 0.07f);
            nebula.setColorAt(0.0, nc);
            nebula.setColorAt(1.0, QColor(0, 0, 0, 0));
            p.fillRect(static_cast<int>(sx - nr * 2.5), static_cast<int>(sy - nr * 2.5),
                       static_cast<int>(nr * 5), static_cast<int>(nr * 5), nebula);
          } else {
            float twink = 0.4f + 0.1f * std::sin(anim_time_ * 2.0f + lx * 0.01f);
            p.setPen(Qt::NoPen);
            p.setBrush(QColor(255, 255, 255, static_cast<int>(twink * 120)));
            p.drawEllipse(QPointF(sx, sy), 0.8, 0.8);
          }
        }
      }
    }
  };

  drawLayer(0.1f, 800.0f, 60, false); // stars
  drawLayer(0.3f, 1500.0f, 2, true);  // nebulae
  
  p.restore();
}

// ─── Star View ────────────────────────────────────────────────────────────────

void GalaxyMapView::drawStarView(QPainter &p) {
  p.save();
  for (int i = 0; i < stars_.size(); ++i) {
    const GalaxyStar &star = stars_[i];
    QPointF sp = worldToScreen(star.position);

    if (sp.x() < -20 || sp.x() > width() + 20 || sp.y() < -20 || sp.y() > height() + 20)
      continue;

    float twinkle = 0.7f + 0.3f * sinf(anim_time_ * 2.5f + star.twinkle_phase);
    // Size: BPM-driven — fast songs are bigger, slow songs are smaller
    float bpm_factor = std::clamp((star.bpm - 60.0f) / 140.0f, 0.0f, 1.0f); // 0 @ 60bpm, 1 @ 200bpm
    float bpm_size = 0.6f + bpm_factor * 1.4f; // range 0.6x..2.0x
    float r = star.base_size * bpm_size * zoom_ * 120.0f * twinkle;
    r = std::max(1.5f, std::min(r, 10.0f));

    QColor sc(static_cast<int>(star.color.x() * 255),
              static_cast<int>(star.color.y() * 255),
              static_cast<int>(star.color.z() * 255));

    // Outer glow
    QRadialGradient glow(sp, r * 3.0);
    glow.setColorAt(0.0, QColor(sc.red(), sc.green(), sc.blue(), 80));
    glow.setColorAt(1.0, QColor(0, 0, 0, 0));
    p.setPen(Qt::NoPen);
    p.setBrush(glow);
    p.drawEllipse(sp, r * 3.0, r * 3.0);

    // Core dot
    QRadialGradient core(sp - QPointF(r * 0.3, r * 0.3), r);
    core.setColorAt(0.0, Qt::white);
    core.setColorAt(0.4, sc);
    core.setColorAt(1.0, sc.darker(150));
    p.setBrush(core);
    p.drawEllipse(sp, r, r);
  }

  // Draw selected ring
  if (selected_star_ >= 0 && selected_star_ < stars_.size()) {
    QPointF sp = worldToScreen(stars_[selected_star_].position);
    float pr = 12.0f + select_pulse_ * 30.0f;
    int alpha = static_cast<int>((1.0f - select_pulse_) * 200.0f);
    p.setPen(QPen(QColor(255, 220, 100, alpha), 2.0));
    p.setBrush(Qt::NoBrush);
    p.drawEllipse(sp, pr, pr);
  }
  p.restore();
}

// ─── Constellation View ────────────────────────────────────────────────────────

void GalaxyMapView::drawConstellationView(QPainter &p) {
  float blend = std::max(0.0f, std::min(1.0f, (zoom_ - kZoomStarView) / (kZoomConstellation - kZoomStarView)));

  p.save();
  p.setRenderHint(QPainter::Antialiasing, true);

  // Draw Vibe Zone corner labels — fixed to screen corners, not world-space
  if (blend > 0.05f) {
    const int pad = 24;
    QFont zoneFont(u"Inter"_s, 13, QFont::Light, true);
    p.setFont(zoneFont);
    QColor ac(255, 255, 255, static_cast<int>(130 * blend));
    p.setPen(ac);

    // Top-left: Distorted / Gritty
    p.drawText(QRectF(pad, pad, 260, 40), Qt::AlignLeft | Qt::AlignVCenter, u"\u2604 Distorted / Gritty"_s);
    // Top-right: Bright / Sharp
    p.drawText(QRectF(width() - 260 - pad, pad, 260, 40), Qt::AlignRight | Qt::AlignVCenter, u"Bright / Sharp \u2600"_s);
    // Bottom-left: Acoustic / Warm
    p.drawText(QRectF(pad, height() - pad - 40, 260, 40), Qt::AlignLeft | Qt::AlignVCenter, u"\u266b Acoustic / Warm"_s);
    // Bottom-right: Tonal / Pure
    p.drawText(QRectF(width() - 260 - pad, height() - pad - 40, 260, 40), Qt::AlignRight | Qt::AlignVCenter, u"Tonal / Pure \u266a"_s);
  }

  // Draw selected star: album art thumbnail + name pill
  if (selected_star_ >= 0 && selected_star_ < stars_.size()) {
    const GalaxyStar &star = stars_[selected_star_];
    QPointF sp = worldToScreen(star.position);

    if (sp.x() >= -100 && sp.x() <= width() + 100 && sp.y() >= -100 && sp.y() <= height() + 100) {
      // Trigger art load if not cached yet
      if (!album_art_cache_.contains(star.album_key) && !album_art_loading_.contains(star.album_key)) {
        loadStarArt(selected_star_);
      }

      float bpm_factor = std::clamp((star.bpm - 60.0f) / 140.0f, 0.0f, 1.0f);
      float dot_r = std::max(1.5f, std::min(star.base_size * (0.6f + bpm_factor * 1.4f) * zoom_ * 120.0f, 10.0f));

      QColor star_col(static_cast<int>(star.color.x() * 255),
                      static_cast<int>(star.color.y() * 255),
                      static_cast<int>(star.color.z() * 255));

      // Album art thumbnail above the star dot
      constexpr float artSz = 64.0f;
      QRectF artRect(sp.x() - artSz / 2.0, sp.y() - dot_r - artSz - 6, artSz, artSz);

      if (album_art_cache_.contains(star.album_key)) {
        p.save();
        QPainterPath clip;
        clip.addRoundedRect(artRect, 8, 8);
        p.setClipPath(clip);
        p.drawPixmap(artRect.toRect(), album_art_cache_[star.album_key]);
        p.restore();
      } else {
        p.setBrush(star_col.darker(200));
        p.setPen(Qt::NoPen);
        p.drawRoundedRect(artRect, 8, 8);
        p.setPen(QColor(255, 255, 255, 80));
        p.setFont(QFont(u"Inter"_s, static_cast<int>(artSz * 0.3)));
        p.drawText(artRect, Qt::AlignCenter, u"\u266a"_s);
      }
      p.setPen(QPen(QColor(star_col.red(), star_col.green(), star_col.blue(), 140), 1.5));
      p.setBrush(Qt::NoBrush);
      p.drawRoundedRect(artRect, 8, 8);

      // Name + artist pill below the star dot
      QFont label_font(u"Inter"_s, 9, QFont::Medium);
      p.setFont(label_font);
      QFontMetrics fm(label_font);
      QString lbl = star.title.isEmpty() ? u"?"_s : star.title;
      QString sub = star.artist;
      int tw = std::max(fm.horizontalAdvance(lbl), fm.horizontalAdvance(sub)) + 16;

      QRectF pill(sp.x() - tw / 2.0, sp.y() + dot_r + 5, tw, 34);
      p.setBrush(QColor(5, 5, 20, 210));
      p.setPen(QPen(QColor(star_col.red(), star_col.green(), star_col.blue(), 120), 1.0));
      p.drawRoundedRect(pill, 6, 6);

      p.setPen(Qt::white);
      p.setFont(QFont(u"Inter"_s, 9, QFont::Bold));
      p.drawText(pill.adjusted(0, 4, 0, 0), Qt::AlignTop | Qt::AlignHCenter, lbl);
      p.setPen(QColor(180, 180, 210));
      p.setFont(QFont(u"Inter"_s, 8));
      p.drawText(pill.adjusted(0, 18, 0, 0), Qt::AlignTop | Qt::AlignHCenter, sub);
    }
  }

  p.restore();
}

// ─── Planet View ──────────────────────────────────────────────────────────────

void GalaxyMapView::drawPlanetView(QPainter &p) {
  p.save();
  for (int i = 0; i < stars_.size(); ++i) {
    const GalaxyStar &star = stars_[i];
    QPointF sp = worldToScreen(star.position);

    float artSize = zoom_ * 45.0f;
    artSize = std::max(20.0f, std::min(artSize, 64.0f));
    float hs = artSize / 2.0f;

    if (sp.x() < -hs - 40 || sp.x() > width() + hs + 40 ||
        sp.y() < -hs - 40 || sp.y() > height() + hs + 40)
      continue;

    // Load high-res art
    if (!album_art_cache_.contains(star.album_key) && !album_art_loading_.contains(star.album_key)) {
      loadStarArt(i);
    }

    bool is_hovered = (i == hovered_star_);
    bool is_selected = (i == selected_star_);

    QColor sc(static_cast<int>(star.color.x() * 255),
              static_cast<int>(star.color.y() * 255),
              static_cast<int>(star.color.z() * 255));

    // --- Halo effect ---
    float haloR = hs * (is_hovered ? 1.6f : 1.3f);
    QRadialGradient halo(sp, haloR);
    QColor hc = sc.lighter(130);
    halo.setColorAt(0.0, QColor(0, 0, 0, 0));
    halo.setColorAt(0.5, QColor(hc.red(), hc.green(), hc.blue(), is_hovered ? 60 : 25));
    halo.setColorAt(1.0, QColor(0, 0, 0, 0));
    p.setPen(Qt::NoPen);
    p.setBrush(halo);
    p.drawEllipse(sp, haloR, haloR);


    // --- Album art ---
    QRectF artRect(sp.x() - hs, sp.y() - hs, artSize, artSize);
    if (album_art_cache_.contains(star.album_key)) {
      p.save();
      QPainterPath clip;
      clip.addRoundedRect(artRect, 8, 8);
      p.setClipPath(clip);
      p.drawPixmap(artRect.toRect(), album_art_cache_[star.album_key]);
      p.restore();

      // Subtle brightness boost on hover
      if (is_hovered) {
        p.save();
        QPainterPath clip2;
        clip2.addRoundedRect(artRect, 8, 8);
        p.setClipPath(clip2);
        p.fillRect(artRect, QColor(255, 255, 255, 25));
        p.restore();
      }
    } else {
      // Placeholder: colored rounded rect
      p.setBrush(sc.darker(200));
      p.setPen(QPen(sc, 2));
      p.drawRoundedRect(artRect, 8, 8);
      // Music note placeholder
      p.setPen(QColor(255, 255, 255, 100));
      p.setFont(QFont(u"Inter"_s, static_cast<int>(artSize * 0.3)));
      p.drawText(artRect, Qt::AlignCenter, u"\u266a"_s);
    }

    // --- Border ring ---
    {
      p.setPen(QPen(QColor(sc.red(), sc.green(), sc.blue(), is_hovered ? 180 : 80), is_hovered ? 2.5 : 1.5));
      p.setBrush(Qt::NoBrush);
      p.drawRoundedRect(artRect, 8, 8);
    }

    // --- Selected pulse ring ---
    if (is_selected) {
      float pr = hs + 6.0f + select_pulse_ * 25.0f;
      int alpha = static_cast<int>((1.0f - select_pulse_) * 255.0f);
      p.setPen(QPen(QColor(255, 220, 100, alpha), 2.5));
      p.setBrush(Qt::NoBrush);
      p.drawEllipse(sp, pr, pr);
    }

    // --- Floating label ---
    if (is_hovered || is_selected) {
      float labelY = sp.y() + hs + 10.0f;
      // Background pill
      QString lbl = star.title.isEmpty() ? u"Unknown"_s : star.title;
      QString lblSub = star.artist;

      QFont titleFont(u"Inter"_s, 11, QFont::Bold);
      QFont subFont(u"Inter"_s, 9);
      QFontMetrics fm(titleFont), fm2(subFont);
      int textW = std::max(fm.horizontalAdvance(lbl), fm2.horizontalAdvance(lblSub)) + 20;
      int textH = 42;

      QRectF pill(sp.x() - textW / 2.0, labelY, textW, textH);
      p.setBrush(QColor(10, 10, 30, 200));
      p.setPen(QPen(QColor(sc.red(), sc.green(), sc.blue(), 120), 1.0));
      p.drawRoundedRect(pill, 8, 8);

      p.setPen(Qt::white);
      p.setFont(titleFont);
      p.drawText(pill.adjusted(0, 4, 0, 0), Qt::AlignTop | Qt::AlignHCenter, lbl);
      p.setPen(QColor(180, 180, 210));
      p.setFont(subFont);
      p.drawText(pill.adjusted(0, 22, 0, 0), Qt::AlignTop | Qt::AlignHCenter, lblSub);
      
      if (is_selected) {
        // (URL intentionally omitted — too long)
      }
    }
  }
  p.restore();
}

// ─── Input ────────────────────────────────────────────────────────────────────

void GalaxyMapView::wheelEvent(QWheelEvent *event) {
  float delta = static_cast<float>(event->angleDelta().y());
  float factor = (delta > 0) ? 1.12f : (1.0f / 1.12f);
  zoom_ = std::max(0.008f, std::min(zoom_ * factor, 8.0f));
  update();
}

void GalaxyMapView::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton) {
    is_dragging_ = true;
    last_mouse_pos_ = event->position();
    velocity_ = QVector2D(0, 0);

    // Only LeftButton interacts with stars
    if (event->button() == Qt::LeftButton) {
      QVector2D worldClick = screenToWorld(event->position());
      float bestDist = 9999.0f;
      int bestIdx = -1;

      for (int i = 0; i < stars_.size(); ++i) {
        // Calculate hit radius based on visual size in screen space
        float visualRadius;
        if (zoom_ < kZoomStarView) {
          visualRadius = std::max(6.0f, stars_[i].base_size * zoom_ * 80.0f);
        } else if (zoom_ < kZoomConstellation) {
          visualRadius = std::max(12.0f, zoom_ * 30.0f);
        } else {
          float artSize = std::max(48.0f, std::min(zoom_ * 150.0f, 256.0f));
          visualRadius = artSize * 0.5f + 10.0f;
        }

        float hitRadiusWorld = visualRadius / zoom_;
        float dist = (stars_[i].position - worldClick).length();
        if (dist < hitRadiusWorld && dist < bestDist) {
          bestDist = dist;
          bestIdx = i;
        }
      }

      if (bestIdx >= 0) {
        // Bring to front
        GalaxyStar clickedStar = stars_.takeAt(bestIdx);
        stars_.append(clickedStar);
        selected_star_ = stars_.size() - 1;
        select_pulse_ = 0.0f;
        update();
      }
    }
  }
}

void GalaxyMapView::mouseDoubleClickEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    QVector2D worldClick = screenToWorld(event->position());
    int bestIdx = -1;
    float bestDist = 9999.0f;

    for (int i = 0; i < stars_.size(); ++i) {
      float visualRadius = (zoom_ < kZoomStarView) ? 15.0f : 40.0f; // Simplified for dblclick
      float hitRadiusWorld = visualRadius / zoom_;
      float dist = (stars_[i].position - worldClick).length();
      if (dist < hitRadiusWorld && dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }

    if (bestIdx >= 0) {
      const GalaxyStar &star = stars_[bestIdx];
      if (star.all_songs_index >= 0 && star.all_songs_index < all_songs_.size()) {
        const Song &song = all_songs_[star.all_songs_index];
        qLog(Debug) << "GalaxyMapView: adding song to playlist:" << song.title() << "URL:" << song.url().toString();
        app_->playlist_manager()->InsertSongsOrCollectionItems(app_->playlist_manager()->current_id(), SongList() << song);
      } else {
        qLog(Error) << "GalaxyMapView: invalid all_songs_index" << star.all_songs_index << "for star" << bestIdx;
      }
    }
  }
}

void GalaxyMapView::mouseMoveEvent(QMouseEvent *event) {
  if (is_dragging_) {
    QPointF diff = event->position() - last_mouse_pos_;
    QVector2D worldDiff(static_cast<float>(diff.x()) / zoom_,
                        static_cast<float>(-diff.y()) / zoom_);
    pan_ -= worldDiff;
    last_mouse_pos_ = event->position();
    velocity_ = worldDiff * -0.5f;
    update();
  }

  // Hover detection (only if not dragging)
  if (!is_dragging_) {
    QVector2D worldMouse = screenToWorld(event->position());
    int prev = hovered_star_;
    hovered_star_ = -1;
    float bestDist = 9999.0f;
    for (int i = 0; i < stars_.size(); ++i) {
      float vRadius = (zoom_ < kZoomStarView) ? 10.0f : 30.0f;
      float hitRadiusWorld = vRadius / zoom_;
      float dist = (stars_[i].position - worldMouse).length();
      if (dist < hitRadiusWorld && dist < bestDist) {
        bestDist = dist;
        hovered_star_ = i;
      }
    }
    if (hovered_star_ != prev) {
      setCursor(hovered_star_ >= 0 ? Qt::PointingHandCursor : Qt::ArrowCursor);
      update();
    }
  }
}

void GalaxyMapView::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton) {
    is_dragging_ = false;
  }
}
