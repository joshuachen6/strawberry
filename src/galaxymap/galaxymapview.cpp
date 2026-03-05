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
#include <QThread>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlRecord>
#include <QVariant>
#include <QProcess>
#include <QStandardPaths>
#include <QFileSystemWatcher>
#include <QDir>

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
      select_pulse_(0.0f),
      db_watcher_(nullptr),
      deep_embedding_process_(nullptr),
      deep_embedding_progress_(-1.0f),
      live_reveal_timer_(nullptr),
      deep_embedding_started_(false),
      deep_embedding_scan_once_(false) {
  setMouseTracking(true);
  setAttribute(Qt::WA_OpaquePaintEvent);
  setMinimumSize(200, 200);

  QObject::connect(timer_, &QTimer::timeout, this, &GalaxyMapView::updateFrame);
  timer_->start(16);  // ~60fps
}

GalaxyMapView::~GalaxyMapView() {
    if (deep_embedding_process_) {
        deep_embedding_process_->kill();
        deep_embedding_process_->waitForFinished(1000);
        delete deep_embedding_process_;
    }
    if (live_reveal_timer_) {
        live_reveal_timer_->stop();
    }
}

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

  // Database Monitoring for Deep Embeddings
  if (!db_watcher_) {
    db_watcher_ = new QFileSystemWatcher(this);
    QString db_dir = QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + QStringLiteral("/strawberry/strawberry");
    QString db_path = db_dir + QStringLiteral("/galaxy_embeddings.db");
    
    QDir().mkpath(db_dir);
    db_watcher_->addPath(db_dir);
    if (QFile::exists(db_path)) {
      db_watcher_->addPath(db_path);
    }
    connect(db_watcher_, &QFileSystemWatcher::fileChanged, this, [this](const QString &path) {
      if (path.endsWith(QStringLiteral("galaxy_embeddings.db"))) {
        if (!deep_embedding_process_ || deep_embedding_process_->state() == QProcess::NotRunning) {
          fetchSongsFromBackend();
        }
      }
    });
    connect(db_watcher_, &QFileSystemWatcher::directoryChanged, this, [this, db_path]() {
      if (QFile::exists(db_path) && !db_watcher_->files().contains(db_path)) {
        db_watcher_->addPath(db_path);
        fetchSongsFromBackend();
      }
    });
  }

  // Step 1: Show placeholder dots immediately so the canvas isn't blank
  buildPlaceholderStars();

  // Step 2: Try loading from already-populated model nodes
  tryLoadFromModel();

  // Step 3: Kick off async backend fetch for full song data
  fetchSongsFromBackend();
}

void GalaxyMapView::ResetScanFlag() {
  if (deep_embedding_process_) {
    deep_embedding_process_->kill();
    deep_embedding_process_->waitForFinished(1000);
  }
  deep_embedding_started_ = false;
  deep_embedding_scan_once_ = false;
  deep_embedding_progress_ = -1.0f;
  fetchSongsFromBackend(true); // Manually requested full scan
}

void GalaxyMapView::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  if (all_songs_.isEmpty()) {
    tryLoadFromModel();
    fetchSongsFromBackend(false);
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
  selected_star_ = -1;
  hovered_star_ = -1;
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
  QMetaObject::invokeMethod(this, [this]() {
    fetchSongsFromBackend(false);
  }, Qt::QueuedConnection);
}

void GalaxyMapView::fetchSongsFromBackend(bool force_scan) {
  CollectionBackend *backend = app_->collection_backend().get();
  // Invoke GetAllSongs on the backend's own thread via queued connection
  QMetaObject::invokeMethod(backend, [this, backend, force_scan]() {
    SongList songs = backend->GetAllSongs();
    qLog(Info) << "GalaxyMapView::fetchSongsFromBackend: got" << songs.size() << "songs";
    // Deliver results to the main thread
    QMetaObject::invokeMethod(this, [this, songs, force_scan]() {
      all_songs_ = songs;
      qLog(Info) << "GalaxyMapView: Building stars for" << songs.size() << "songs";
      buildStars(songs, force_scan);
    }, Qt::QueuedConnection);
  }, Qt::QueuedConnection);
}

void GalaxyMapView::DeepEmbeddings(bool force_scan) {
  QMetaObject::invokeMethod(this, [this, force_scan]() {
      // Logic:
      // 1. If force_scan is true, always start.
      // 2. If already scanning, DON'T start again.
      // 3. If we've already done ONE scan this session, DON'T auto-restart (unless force_scan).
      if ((deep_embedding_started_ || deep_embedding_scan_once_) && !force_scan) return;
      
      deep_embedding_started_ = true;
      // Clear flag if we are starting a manual force_scan
      if (force_scan) deep_embedding_scan_once_ = false;

      if (deep_embedding_process_) {
          deep_embedding_process_->kill();
          deep_embedding_process_->waitForFinished(500);
          delete deep_embedding_process_;
      }
      deep_embedding_process_ = new QProcess(this);
      QString music_dir = QStandardPaths::writableLocation(QStandardPaths::MusicLocation);
      QString script_path = QStringLiteral("/home/jchen/Documents/Projects/strawberry/src/galaxymap/analyze_library.py");
      
      QString python_path = QStringLiteral("python3");
      if (QFile::exists(QStringLiteral("/home/jchen/Documents/Projects/strawberry/build/.venv/bin/python3"))) {
          python_path = QStringLiteral("/home/jchen/Documents/Projects/strawberry/build/.venv/bin/python3");
      }
      deep_embedding_process_->setProgram(python_path);
      QStringList args = {script_path, QStringLiteral("--dir"), music_dir};
      if (force_scan) args << QStringLiteral("--force");
      deep_embedding_process_->setArguments(args);
      
      connect(deep_embedding_process_, &QProcess::readyReadStandardOutput, this, [this]() {
          while (deep_embedding_process_->canReadLine()) {
              QByteArray line = deep_embedding_process_->readLine().trimmed();
              if (line.isEmpty()) continue;
              
              QJsonDocument doc = QJsonDocument::fromJson(line);
              if (doc.isObject()) {
                  QJsonObject obj = doc.object();
                  if (obj.contains(QStringLiteral("progress"))) {
                      deep_embedding_progress_ = obj.value(QStringLiteral("progress")).toDouble();
                      deep_embedding_status_ = obj.value(QStringLiteral("status")).toString();
                      update();
                  }
              }
          }
      });

      connect(deep_embedding_process_, &QProcess::readyReadStandardError, this, [this]() {
          qWarning() << "PYTHON ERROR:" << deep_embedding_process_->readAllStandardError();
      });
      
      connect(deep_embedding_process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int, QProcess::ExitStatus) {
          deep_embedding_started_ = false;
          deep_embedding_scan_once_ = true; // Mark as done for this map session
          deep_embedding_progress_ = 1.0f;
          deep_embedding_status_ = tr("Scan finished");
          
          if (live_reveal_timer_) {
              live_reveal_timer_->stop();
          }

          // Hide progress bar after 3 seconds
          QTimer::singleShot(3000, this, [this]() {
            if (!deep_embedding_started_) {
              deep_embedding_progress_ = -1.0f;
              update();
            }
          });

          // Final database fetch WITHOUT triggering a restart
          fetchSongsFromBackend(false);
          update();
      });
      
      deep_embedding_progress_ = 0.0f;
      deep_embedding_process_->start();

      if (!live_reveal_timer_) {
          live_reveal_timer_ = new QTimer(this);
          connect(live_reveal_timer_, &QTimer::timeout, this, [this]() { fetchSongsFromBackend(false); });
      }
      live_reveal_timer_->start(5000);
  }, Qt::QueuedConnection);
}

void GalaxyMapView::buildStars(const SongList &songs, bool force_scan) {
  // Version sentinel — clear stale cache if math changed
  const int kVibeVersion = 13;
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

    QColor tc; tc.setRgbF(std::clamp(normR, 0.0, 1.0), std::clamp(normG, 0.0, 1.0), std::clamp(normB, 0.0, 1.0));
    if (!tc.isValid()) tc = QColor(100, 150, 255);
    float h, s, v; tc.getHsvF(&h, &s, &v);
    
    // Defensive clamping for setHsvF
    h = std::isnan(h) ? 0.0f : std::clamp(h, 0.0f, 1.0f);
    s = std::isnan(static_cast<float>(saturation)) ? 0.0f : std::clamp(static_cast<float>(saturation), 0.0f, 1.0f);
    v = std::isnan(v) ? 0.8f : std::clamp(v, 0.0f, 1.0f);
    
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
      DeepEmbeddings(force_scan);
  }

  auto future_ = QtConcurrent::run([this, songs, raw, jitter_seeds, processMoodMath, backend_type]() {
    struct SongEntry { QVector2D pos; QVector3D col; float bpm; QString genre; };
    QVector<SongEntry> entries;
    entries.reserve(songs.size());
    QRandomGenerator bg_rng(0xDEADBEEF);
    QSettings vibe_cache(u"Strawberry"_s, u"GalaxyVibeMap"_s);

    struct DbEntry { QVector2D pos; QVector3D col; QString genre; };
    QHash<QString, DbEntry> db_embeddings;
    if (backend_type == AppearanceSettings::GalaxyBackendType::DeepEmbeddings) {
      QString db_path = QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + QStringLiteral("/strawberry/strawberry/galaxy_embeddings.db");
      if (QFile::exists(db_path)) {
        QString connection_name = QStringLiteral("embeddings_db_") + QString::number(bg_rng.generate());
        {
          QSqlDatabase db = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connection_name);
          db.setDatabaseName(db_path);
          if (db.open()) {
            QSqlQuery q(db);
            if (!q.exec(QStringLiteral("SELECT path, x, y, vibrancy, virtual_genre, confidence_score FROM embeddings"))) {
              q.exec(QStringLiteral("SELECT path, x, y, vibrancy, virtual_genre FROM embeddings"));
            }
            int count = 0;
            while (q.next()) {
              count++;
              QString path = q.value(0).toString();
              float vib = q.value(3).toFloat();
              QString v_genre = q.value(4).toString();
              float conf = q.record().indexOf(u"confidence_score"_s) >= 0 ? q.value(5).toFloat() : 1.0f;

              if (std::isnan(vib)) vib = 0.0f;
              if (std::isnan(conf)) conf = 0.0f;
              
              int hue = 0;
              if      (v_genre.contains(u"Aggressive"_s)) hue = 0;   // Crimson
              else if (v_genre.contains(u"Metal"_s))      hue = 12;  // Rust
              else if (v_genre.contains(u"Rock"_s))       hue = 32;  // Amber
              else if (v_genre.contains(u"Punk"_s))       hue = 18;  // Blood Orange
              else if (v_genre.contains(u"Hip-Hop"_s))    hue = 58;  // Gold
              else if (v_genre.contains(u"Soul"_s))       hue = 345; // Raspberry
              else if (v_genre.contains(u"Disco"_s))      hue = 310; // Magenta
              else if (v_genre.contains(u"Reggae"_s))     hue = 120; // Grass Green
              else if (v_genre.contains(u"DnB"_s))        hue = 100; // Acid Green
              else if (v_genre.contains(u"Hardstyle"_s)) hue = 5;   // Bright Fire Red
              else if (v_genre.contains(u"Techno"_s))     hue = 165; // Emerald
              else if (v_genre.contains(u"Electronic"_s)) hue = 195; // Cyan
              else if (v_genre.contains(u"Synthwave"_s))  hue = 305; // Neon Purple
              else if (v_genre.contains(u"Ambient"_s))    hue = 215; // Cobalt
              else if (v_genre.contains(u"Cinematic"_s))  hue = 200; // Sky Blue
              else if (v_genre.contains(u"Classical"_s))  hue = 265; // Amethyst
              else if (v_genre.contains(u"Jazz"_s))       hue = 295; // Deep Lavender 
              else if (v_genre.contains(u"Pop"_s))        hue = 330; // Rose
              else if (v_genre.contains(u"Acoustic"_s))   hue = 45;  // Peach
              else if (v_genre.contains(u"Folk"_s))       hue = 38;  // Earthy Orange
              else if (v_genre.contains(u"Lo-Fi"_s))      hue = 280; // Muted Violet
              else if (v_genre.contains(u"Blues"_s))      hue = 230; // Deep Sky Blue 
              else if (v_genre.contains(u"Industrial"_s)) hue = 180; // Cold Teal
              else hue = static_cast<int>(qHash(v_genre) % 360);

              // Low confidence or "Unclassified" -> Muted / Grey
              int sat = std::max(160, std::min(255, 160 + static_cast<int>(vib * 100)));
              int val = std::max(220, std::min(255, 200 + static_cast<int>(vib * 120)));

              if (v_genre == u"Unclassified"_s) {
                sat = 40; // Very desaturated
                val = 150; // Dimmed
                hue = 0; 
              }

              QColor tc;
              int safe_hue = ((hue % 360) + 360) % 360;
              tc.setHsv(safe_hue, std::clamp(sat, 0, 255), std::clamp(val, 0, 255));
              
              QVector3D col(tc.redF(), tc.greenF(), tc.blueF());
              if (q.value(1).isNull()) {
                float fx = (static_cast<float>(bg_rng.generateDouble()) - 0.5f) * 1500.0f;
                float fy = (static_cast<float>(bg_rng.generateDouble()) - 0.5f) * 1500.0f;
                db_embeddings[path] = {QVector2D(fx, fy), col, v_genre};
              } else {
                float x = q.value(1).toFloat();
                float y = q.value(2).toFloat();
                db_embeddings[path] = {QVector2D(x, y), col, v_genre};
              }
            }
            qLog(Info) << "GalaxyMapView: Loaded" << count << "embeddings from DB";
            db.close();
          }
        }
        QSqlDatabase::removeDatabase(connection_name);
      }
    }

    for (int i = 0; i < songs.size(); ++i) {
      const Song &song = songs[i];
      const RawEntry &re = raw[i];

      if (!song.is_valid() || song.unavailable()) {
        entries.append({QVector2D(0,0), QVector3D(0.5f,0.5f,0.5f), 120.0f, u"Unknown"_s});
        continue;
      }

      float bpm = song.bpm() > 0.0f ? song.bpm() : 120.0f;
      QString hk = u"vcode_"_s + QString::fromUtf8(song.url().toEncoded().toHex());
      
      if (backend_type == AppearanceSettings::GalaxyBackendType::DeepEmbeddings) {
        QString local_path = song.url().toLocalFile();
        if (db_embeddings.contains(local_path)) {
          DbEntry emb = db_embeddings.value(local_path);
          entries.append({emb.pos, emb.col, bpm, emb.genre});
          continue;
        } else {
          // Fallback if not processed yet or not local
          float x = (static_cast<float>(bg_rng.generateDouble()) - 0.5f) * 1500.0f;
          float y = (static_cast<float>(bg_rng.generateDouble()) - 0.5f) * 1500.0f;
          entries.append({QVector2D(x, y), QVector3D(0.5f,0.5f,0.5f), bpm, u"Unknown"_s});
          continue;
        }
      }

      if (re.from_cache) {
        entries.append({QVector2D(re.cached_x, re.cached_y), QVector3D(re.cached_r, re.cached_g, re.cached_b), bpm, u"Unknown"_s});
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
      entries.append({QVector2D(x, mood_y), col, bpm, u"Unknown"_s});
    }

    // ── PHASE 2 (main thread): build star objects immediately ─────────────────
    QMetaObject::invokeMethod(this, [this, songs, entries, jitter_seeds]() {
      stars_.clear();
      selected_star_ = -1;
      hovered_star_ = -1;

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
        star.color         = entries[i].col;
        star.base_size     = 3.0f + std::min(1.0f, std::max(0.0f, song.rating()))*2.0f;
        star.twinkle_phase = static_cast<float>(rng.generateDouble()*2.0*M_PI);
        star.title         = song.title();
        star.artist        = song.artist();
        star.album         = song.album();
        star.album_artist  = song.effective_albumartist();
        star.bpm           = bpm;
        star.song_id       = song.id();
        star.all_songs_index = i;
        star.genre         = entries[i].genre;
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
        (void)QtConcurrent::run([this, i, url]() {
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
  album_art_thumb_cache_[album_key] = pixmap.scaled(64, 64, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);

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
  
  // Progress Overlay
  if (deep_embedding_progress_ >= 0.0f) {
      int bar_w = 300;
      int bar_h = 24;
      int cx = width() / 2;
      int cy = height() - 60;
      p.fillRect(cx - bar_w/2, cy, bar_w, bar_h, QColor(0, 0, 0, 150));
      p.fillRect(cx - bar_w/2, cy, static_cast<int>(bar_w * deep_embedding_progress_), bar_h, QColor(80, 150, 255, 200));
      p.setPen(Qt::white);
      p.drawRect(cx - bar_w/2, cy, bar_w, bar_h);
      
      p.drawText(cx - bar_w/2, cy + bar_h + 16, deep_embedding_status_);
      
      QString prog_text = QStringLiteral("Scanning Galaxy... %1%").arg(static_cast<int>(deep_embedding_progress_ * 100));
      p.drawText(QRect(cx - bar_w/2, cy, bar_w, bar_h), Qt::AlignCenter, prog_text);
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
      
    // Preload art
    loadStarArt(i);

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
      QFont genre_font(u"Inter"_s, 8, QFont::Normal);
      p.setFont(label_font);
      QFontMetrics fm(label_font), fmG(genre_font);
      QString lbl = star.title.isEmpty() ? u"?"_s : star.title;
      QString sub = star.artist;
      QString gen = star.genre.isEmpty() ? u"Unknown"_s : star.genre;

      int tw = std::max({fm.horizontalAdvance(lbl), fm.horizontalAdvance(sub), fmG.horizontalAdvance(gen)}) + 20;

      QRectF pill(sp.x() - tw / 2.0, sp.y() + dot_r + 5, tw, 50);
      p.setBrush(QColor(5, 5, 20, 210));
      p.setPen(QPen(QColor(star_col.red(), star_col.green(), star_col.blue(), 120), 1.0));
      p.drawRoundedRect(pill, 6, 6);

      p.setPen(Qt::white);
      p.setFont(QFont(u"Inter"_s, 10, QFont::Bold));
      p.drawText(pill.adjusted(0, 4, 0, 0), Qt::AlignTop | Qt::AlignHCenter, lbl);
      p.setPen(QColor(180, 180, 210));
      p.setFont(QFont(u"Inter"_s, 9));
      p.drawText(pill.adjusted(0, 18, 0, 0), Qt::AlignTop | Qt::AlignHCenter, sub);
      p.setPen(QColor(150, 150, 180));
      p.setFont(genre_font);
      p.drawText(pill.adjusted(0, 33, 0, 0), Qt::AlignTop | Qt::AlignHCenter, gen);
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
      QString lblGenre = star.genre.isEmpty() ? u"Unknown"_s : star.genre;

      QFont titleFont(u"Inter"_s, 11, QFont::Bold);
      QFont subFont(u"Inter"_s, 9);
      QFont genreFont(u"Inter"_s, 8, QFont::Normal);
      QFontMetrics fm(titleFont), fm2(subFont), fmG(genreFont);
      int textW = std::max({fm.horizontalAdvance(lbl), fm2.horizontalAdvance(lblSub), fmG.horizontalAdvance(lblGenre)}) + 20;
      int textH = 58;

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
      p.setPen(QColor(150, 150, 180));
      p.setFont(genreFont);
      p.drawText(pill.adjusted(0, 38, 0, 0), Qt::AlignTop | Qt::AlignHCenter, lblGenre);
      
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
    press_start_pos_ = event->position();
    velocity_ = QVector2D(0, 0);
  }
}

void GalaxyMapView::mouseDoubleClickEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    QPointF clickPos = event->position();
    int bestIdx = -1;
    float bestDistSq = 999999.0f;

    for (int i = 0; i < stars_.size(); ++i) {
      float vRadius;
      if (zoom_ < kZoomStarView) vRadius = 15.0f;
      else if (zoom_ < kZoomConstellation) vRadius = 30.0f;
      else vRadius = 50.0f;

      QPointF starScreen = worldToScreen(stars_[i].position);
      float dx = starScreen.x() - clickPos.x();
      float dy = starScreen.y() - clickPos.y();
      float distSq = dx*dx + dy*dy;

      if (distSq < (vRadius * vRadius) && distSq < bestDistSq) {
        bestDistSq = distSq;
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
    QPointF mousePos = event->position();
    int prev = hovered_star_;
    hovered_star_ = -1;
    float bestDistSq = 999999.0f;
    for (int i = 0; i < stars_.size(); ++i) {
      float vRadius;
      if (zoom_ < kZoomStarView) {
        vRadius = std::max(10.0f, stars_[i].base_size * zoom_ * 100.0f);
      } else if (zoom_ < kZoomConstellation) {
        vRadius = std::max(20.0f, zoom_ * 40.0f);
      } else {
        float artSize = std::max(48.0f, std::min(zoom_ * 150.0f, 256.0f));
        vRadius = artSize * 0.5f + 25.0f;
      }

      QPointF starScreen = worldToScreen(stars_[i].position);
      float dx = starScreen.x() - mousePos.x();
      float dy = starScreen.y() - mousePos.y();
      float distSq = dx*dx + dy*dy;

      if (distSq < (vRadius * vRadius) && distSq < bestDistSq) {
        bestDistSq = distSq;
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

    // Selection logic only if it was a clean click (hardly any movement)
    if (event->button() == Qt::LeftButton) {
      float moveDist = QLineF(press_start_pos_, event->position()).length();
      if (moveDist < 5.0f) {
        QPointF clickPos = event->position();
        float bestDistSq = 999999.0f;
        int bestIdx = -1;

        for (int i = 0; i < stars_.size(); ++i) {
          float visualRadius;
          if (zoom_ < kZoomStarView) {
            visualRadius = std::max(10.0f, stars_[i].base_size * zoom_ * 100.0f);
          } else if (zoom_ < kZoomConstellation) {
            visualRadius = std::max(20.0f, zoom_ * 40.0f);
          } else {
            float artSize = std::max(48.0f, std::min(zoom_ * 150.0f, 256.0f));
            visualRadius = artSize * 0.5f + 25.0f;
          }

          QPointF starScreen = worldToScreen(stars_[i].position);
          float dx = starScreen.x() - clickPos.x();
          float dy = starScreen.y() - clickPos.y();
          float distSq = dx*dx + dy*dy;
          
          if (distSq < (visualRadius * visualRadius) && distSq < bestDistSq) {
            bestDistSq = distSq;
            bestIdx = i;
          }
        }

        if (bestIdx >= 0) {
          // Select and bring to front
          GalaxyStar clickedStar = stars_.takeAt(bestIdx);
          stars_.append(clickedStar);
          selected_star_ = stars_.size() - 1;
          select_pulse_ = 0.0f;
        } else {
          // Clicked empty space — deselect
          selected_star_ = -1;
        }
        update();
      }
    }
  }
}
