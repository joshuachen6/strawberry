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

static float getAubioBPM(const QUrl& url) {
    if (!url.isLocalFile()) return 120.0f;
    QString path = url.toLocalFile();
    
    // Quick in-memory cache to avoid recalculating every time we rebuild map
    static QHash<QString, float> s_bpmCache;
    if (s_bpmCache.contains(path)) return s_bpmCache.value(path);

    uint_t samplerate = 0; 
    uint_t hop_size = 256;
    
    QByteArray pathBa = path.toUtf8();
    aubio_source_t * source = new_aubio_source(pathBa.constData(), samplerate, hop_size);
    if (!source) {
        s_bpmCache.insert(path, 120.0f);
        return 120.0f;
    }
    
    samplerate = aubio_source_get_samplerate(source);
    if (samplerate == 0) samplerate = 44100;
    
    aubio_tempo_t * tempo = new_aubio_tempo("default", 1024, hop_size, samplerate);
    if (!tempo) {
        del_aubio_source(source);
        s_bpmCache.insert(path, 120.0f);
        return 120.0f;
    }
    
    fvec_t * in = new_fvec(hop_size);
    fvec_t * out = new_fvec(2);
    
    // Seek to 30s to avoid intro and get to the beat
    aubio_source_seek(source, samplerate * 30);
    
    uint_t read = 0;
    // Read up to 20 seconds of audio to get a reading
    uint_t max_blocks = (20 * samplerate) / hop_size;
    uint_t blocks = 0;
    
    do {
        aubio_source_do(source, in, &read);
        aubio_tempo_do(tempo, in, out);
        blocks++;
    } while (read == hop_size && blocks < max_blocks);
    
    float bpm = aubio_tempo_get_bpm(tempo);
    
    del_aubio_tempo(tempo);
    del_aubio_source(source);
    del_fvec(in);
    del_fvec(out);
    
    if (bpm <= 0.0f || std::isinf(bpm) || std::isnan(bpm)) {
        bpm = 120.0f;
    }
    
    s_bpmCache.insert(path, bpm);
    return bpm;
}
#else
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

  QRandomGenerator rng(0xDEADBEEF);
  
  QVector<QVector2D> base_positions;
  base_positions.reserve(songs.size());
  
  QVector<float> all_x;
  QVector<float> all_y;
  all_x.reserve(songs.size());
  all_y.reserve(songs.size());

  // Pass 1: Compute raw metrics
  for (int i = 0; i < songs.size(); ++i) {
    const Song &song = songs[i];
    if (!song.is_valid() || song.unavailable()) {
      base_positions.append(QVector2D(0, 0));
      continue;
    }

    float x = 0.0f;
    float mood_y = 0.0f;
    bool has_acoustics = false;

#ifdef HAVE_MOODBAR
    if (app_->moodbar_loader()) {
      MoodbarLoader::LoadResult res = app_->moodbar_loader()->Load(song.url(), song.has_cue());
      if (res.status == MoodbarLoader::LoadStatus::Loaded && !res.data.isEmpty()) {
        const QByteArray &data = res.data;
        int n = data.size() / 3;
        if (n > 0) {
          double rSum = 0, gSum = 0, bSum = 0;
          for (int j = 0; j < n; ++j) {
            rSum += static_cast<unsigned char>(data[j * 3]);
            gSum += static_cast<unsigned char>(data[j * 3 + 1]);
            bSum += static_cast<unsigned char>(data[j * 3 + 2]);
          }
          // Average the sums
          rSum /= n; gSum /= n; bSum /= n;
          
          // Higher overall brightness means higher energy
          double energy = rSum + gSum + bSum;
          
          // Calculate a "pseudo spectral centroid".
          // If R=low, G=mid, B=high (roughly), centroid is weighted sum:
          double centroid = (1.0 * rSum + 2.0 * gSum + 3.0 * bSum) / (energy + 0.1);
          
          // Calculate "flatness" or "richness" based on variance
          double vari_r = 0, vari_g = 0, vari_b = 0;
          for (int j = 0; j < n; ++j) {
            vari_r += std::pow((static_cast<unsigned char>(data[j * 3]) - rSum), 2);
            vari_g += std::pow((static_cast<unsigned char>(data[j * 3 + 1]) - gSum), 2);
            vari_b += std::pow((static_cast<unsigned char>(data[j * 3 + 2]) - bSum), 2);
          }
          vari_r /= n; vari_g /= n; vari_b /= n;
          
          double variance = vari_r + vari_g + vari_b;
          double flatness = variance / (energy + 1.0); // Rough approximation of tonality vs noise
          
          // Combine BPM (if available) with Energy to determine Radius
          float bpm = song.bpm() > 0 ? song.bpm() : getAubioBPM(song.url());
          float base_radius = (energy * 1.5f) + (bpm * 1.5f);
          
          // Angle determined by musical tone (Centroid + Flatness)
          float theta = (centroid * 15.0f) + (flatness * 0.005f);
          
          x = base_radius * 4.0f * std::cos(theta);
          mood_y = base_radius * 4.0f * std::sin(theta);
          
          if (std::isnan(x) || std::isinf(x)) x = 0.0f;
          if (std::isnan(mood_y) || std::isinf(mood_y)) mood_y = 0.0f;
          
          has_acoustics = true;
        }
      }
    }
#endif

    if (!has_acoustics) {
      float length_sec = static_cast<float>(song.length_nanosec()) / 1e9f;
      float bpm = song.bpm() > 0 ? song.bpm() : getAubioBPM(song.url());
      float r = bpm * 4.0f + length_sec * 0.5f;
      float theta = ((length_sec / 360.0f) * M_PI * 2.0f) + (bpm * 0.01f);
      x = r * std::cos(theta);
      mood_y = r * std::sin(theta);
    }

    base_positions.append(QVector2D(x, mood_y));
  }

  // Pass 2: Calculate medians for centering
  for (int i = 0; i < songs.size(); ++i) {
    if (!songs[i].is_valid() || songs[i].unavailable()) continue;
    all_x.append(base_positions[i].x());
    all_y.append(base_positions[i].y());
  }

  float med_x = 0.0f;
  float med_y = 0.0f;
  if (!all_x.isEmpty()) {
    std::sort(all_x.begin(), all_x.end());
    std::sort(all_y.begin(), all_y.end());
    med_x = all_x[all_x.size() / 2];
    med_y = all_y[all_y.size() / 2];
  }

  // Pass 3: Construct stars with medians and generous jitter
  for (int i = 0; i < songs.size(); ++i) {
    const Song &song = songs[i];
    if (!song.is_valid() || song.unavailable()) continue;

    uint h = qHash(song.url().toEncoded());
    auto jitter = [&](uint seed, float range) {
      return (static_cast<float>(qHash(h ^ seed) % 10000) / 10000.0f - 0.5f) * range;
    };

    float x = base_positions[i].x() - med_x + jitter(0x1234, 150.0f);
    float y = base_positions[i].y() - med_y + jitter(0x5678, 150.0f);

    float hue = static_cast<float>(qHash(song.effective_albumartist().toLower()) % 360) / 360.0f;
    float sat = std::clamp(0.6f + std::min(1.0f, std::max(0.0f, song.rating())) * 0.2f, 0.0f, 1.0f);
    QColor col = QColor::fromHsvF(static_cast<double>(hue), static_cast<double>(sat), 1.0);
    if (!col.isValid()) col = QColor(100, 150, 255);

    float bpm = song.bpm();
    if (bpm <= 0.0f) {
      QString group_key = song.album().isEmpty() ? song.effective_albumartist() : song.album();
      bpm = static_cast<float>(qHash(group_key.toLower()) % 120) + 60.0f;
    }

    GalaxyStar star;
    star.position      = QVector2D(x, y);
    star.color         = QVector3D(col.redF(), col.greenF(), col.blueF());
    star.base_size     = 3.0f + std::min(1.0f, std::max(0.0f, song.rating())) * 2.0f;
    star.twinkle_phase = static_cast<float>(rng.generateDouble() * 2.0 * M_PI);
    star.title         = song.title();
    star.artist        = song.artist();
    star.album         = song.album();
    star.album_artist  = song.effective_albumartist();
    star.bpm           = bpm;
    star.song_id       = song.id();
    star.all_songs_index = i;

    star.album_key = song.AlbumKey();
    if (star.album_key.isEmpty()) star.album_key = song.artist() + u"|" + song.album();

    if (!song.art_manual().isEmpty())         star.art_url = song.art_manual();
    else if (!song.art_automatic().isEmpty()) star.art_url = song.art_automatic();

    star.url = song.url();

    stars_.append(star);
  }

  // Pass 4: Build stable constellations (clustering by distance for each artist)
  constellations_.clear();
  QHash<QString, QVector<int>> artist_groups;
  for (int i = 0; i < stars_.size(); ++i) {
    QString artistKey = stars_[i].album_artist;
    if (artistKey.isEmpty()) artistKey = stars_[i].artist;
    if (artistKey.isEmpty()) artistKey = u"Unknown"_s;
    artist_groups[artistKey].append(i);
  }

  float maxDistWorld = 850.0f; // Increased to 850.0f so jitter doesn't disconnect related stars

  struct Edge {
    int i, j;
    float distSq;
    bool operator<(const Edge& o) const { return distSq < o.distSq; }
  };

  for (auto it = artist_groups.begin(); it != artist_groups.end(); ++it) {
    const QVector<int>& indices = it.value();
    
    QVector<Edge> candidate_edges;
    for (int i = 0; i < indices.size(); ++i) {
      for (int j = i + 1; j < indices.size(); ++j) {
        float distSq = (stars_[indices[i]].position - stars_[indices[j]].position).lengthSquared();
        if (distSq < maxDistWorld * maxDistWorld) {
          candidate_edges.append({i, j, distSq});
        }
      }
    }
    std::sort(candidate_edges.begin(), candidate_edges.end());
    
    QVector<int> parent(indices.size());
    for (int i = 0; i < indices.size(); ++i) parent[i] = i;
    
    auto find = [&](int i) {
      int root = i;
      while (root != parent[root]) root = parent[root];
      int curr = i;
      while (curr != root) { int nxt = parent[curr]; parent[curr] = root; curr = nxt; }
      return root;
    };

    QVector<QPair<int, int>> mst_edges;
    for (const Edge& e : candidate_edges) {
      int rootI = find(e.i);
      int rootJ = find(e.j);
      if (rootI != rootJ) {
        parent[rootI] = rootJ;
        mst_edges.append(qMakePair(indices[e.i], indices[e.j]));
      }
    }

    QHash<int, QVector<int>> sub_clusters;
    for (int i = 0; i < indices.size(); ++i) {
      sub_clusters[find(i)].append(indices[i]);
    }

    // Determine the "True Name" of the artist for the label. If they vary inside the group, we just take the first one or it.key().
    // We already use the key, but the key might be lowercased or something? No, it's just the exact string.
    QString lbl = it.key();

    for (auto sub_it = sub_clusters.begin(); sub_it != sub_clusters.end(); ++sub_it) {
      const QVector<int>& cluster = sub_it.value();
      QVector2D centroid(0, 0);
      for (int idx : cluster) centroid += stars_[idx].position;
      centroid /= cluster.size();
      
      QVector<QPair<int, int>> cluster_edges;
      for (const auto& edge : mst_edges) {
        if (cluster.contains(edge.first)) {
           cluster_edges.append(edge);
        }
      }
      
      constellations_.append({cluster, cluster_edges, centroid, lbl});
    }
  }

  update();
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
            float hue = static_cast<float>(rng.generateDouble());
            int nr = 80 + static_cast<int>(rng.generateDouble() * 200);
            QRadialGradient nebula(QPointF(sx, sy), nr * 2.5f);
            QColor nc = QColor::fromHsvF(double(hue), 0.7, 0.6, 0.07);
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
    float r = star.base_size * zoom_ * 120.0f * twinkle;
    r = std::max(1.5f, std::min(r, 8.0f));

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

  for (const GalaxyConstellation &c : constellations_) {
    QPointF centerSp = worldToScreen(c.centroid);
    
    // Quick bounds check: approximate visibility (constellation could be wide)
    if (centerSp.x() < -2000 || centerSp.x() > width() + 2000 ||
        centerSp.y() < -2000 || centerSp.y() > height() + 2000)
      continue;
      
    // Map points and check detailed visibility
    QVector<QPointF> screenPoints;
    screenPoints.reserve(c.star_indices.size());
    bool any_visible = false;
    
    for (int idx : c.star_indices) {
      QPointF sp = worldToScreen(stars_[idx].position);
      screenPoints.append(sp);
      if (sp.x() >= -80 && sp.x() <= width() + 80 && sp.y() >= -80 && sp.y() <= height() + 80)
        any_visible = true;
    }
    
    if (!any_visible) continue;

    // Draw connecting lines
    if (blend > 0.1f && c.star_indices.size() > 1) {
      QColor ac(static_cast<int>(stars_[c.star_indices.first()].color.x() * 255),
                static_cast<int>(stars_[c.star_indices.first()].color.y() * 255),
                static_cast<int>(stars_[c.star_indices.first()].color.z() * 255),
                static_cast<int>(100 * blend));
      p.setPen(QPen(ac, 1.5, Qt::DashLine));
      
      for (const auto& edge : c.edges) {
        QPointF p1 = worldToScreen(stars_[edge.first].position);
        QPointF p2 = worldToScreen(stars_[edge.second].position);
        p.drawLine(p1, p2);
      }
    }

    // Draw stars and hover states
    for (int i = 0; i < c.star_indices.size(); ++i) {
      int idx = c.star_indices[i];
      const GalaxyStar &star = stars_[idx];
      QPointF sp = screenPoints[i];
      
      if (sp.x() < -80 || sp.x() > width() + 80 || sp.y() < -80 || sp.y() > height() + 80)
        continue;
        
      float twinkle = 0.85f + 0.15f * sinf(anim_time_ * 2.5f + star.twinkle_phase);

      float r = star.base_size * std::max(6.0f, zoom_ * 30.0f) * twinkle;
      r = std::min(r, 12.0f);
      
      QColor sc(static_cast<int>(star.color.x() * 255),
                static_cast<int>(star.color.y() * 255),
                static_cast<int>(star.color.z() * 255));
                
      QRadialGradient core(sp, r);
      core.setColorAt(0.0, Qt::white);
      core.setColorAt(0.5, sc);
      core.setColorAt(1.0, sc.darker(150));
      
      p.setPen(Qt::NoPen);
      p.setBrush(core);
      p.drawEllipse(sp, r, r);

      if (idx == selected_star_) {
        float pr = r + 6.0f + select_pulse_ * 15.0f;
        int alpha = static_cast<int>((1.0f - select_pulse_) * 220.0f);
        p.setPen(QPen(QColor(255, 220, 100, alpha), 2.0));
        p.setBrush(Qt::NoBrush);
        p.drawEllipse(sp, pr, pr);
      }

      if (idx == hovered_star_) {
        p.setPen(QColor(255, 255, 255, 200));
        p.setFont(QFont(u"Inter"_s, 9, QFont::Bold));
        p.drawText(QPointF(sp.x() + r + 8.0, sp.y() + 4.0), star.title);
        
        p.setFont(QFont(u"Inter"_s, 8));
        p.setPen(QColor(200, 200, 200, 160));
        QString sublbl = star.artist;
        if (!star.album.isEmpty()) sublbl += u" / " + star.album;
        p.drawText(QPointF(sp.x() + r + 8.0, sp.y() + 18.0), sublbl);
      }
    }
    
    // Draw constellation label
    if (blend > 0.05f && c.star_indices.size() > 1) {
      if (centerSp.x() > -200 && centerSp.x() < width() + 200 && centerSp.y() > -100 && centerSp.y() < height() + 100) {
        QColor ac(static_cast<int>(stars_[c.star_indices.first()].color.x() * 255),
                  static_cast<int>(stars_[c.star_indices.first()].color.y() * 255),
                  static_cast<int>(stars_[c.star_indices.first()].color.z() * 255),
                  static_cast<int>(180 * blend));
                  
        p.setPen(ac);
        p.setFont(QFont(u"Inter"_s, 11, QFont::DemiBold));
        p.drawText(QRectF(centerSp.x() - 150, centerSp.y() - 30, 300, 60), Qt::AlignCenter, c.label);
      }
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

    float artSize = zoom_ * 150.0f;
    artSize = std::max(48.0f, std::min(artSize, 256.0f));
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
        QString urlStr = star.url.toString();
        p.setFont(QFont(u"Inter"_s, 7));
        p.setPen(QColor(150, 150, 180, 150));
        p.drawText(pill.adjusted(4, 34, -4, 0), Qt::AlignBottom | Qt::AlignHCenter, urlStr);
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
