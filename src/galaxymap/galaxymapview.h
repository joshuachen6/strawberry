#ifndef GALAXYMAPVIEW_H
#define GALAXYMAPVIEW_H

#include <QWidget>
#include <QMatrix4x4>
#include <QVector2D>
#include <QVector3D>
#include <QTimer>
#include <QPointF>
#include <QVector>
#include <QMap>
#include <QSet>
#include <QPixmap>
#include <QColor>
#include <QUrl>
#include <QString>
#include "core/song.h"

#include <QProcess>
#include <QFileSystemWatcher>

class Application;
class AlbumCoverLoaderResult;

// A single star in the galaxy map
struct GalaxyStar {
  GalaxyStar() : position(0, 0), color(1, 1, 1), base_size(3.0f), twinkle_phase(0.0f), song_id(-1), bpm(0.0f), all_songs_index(-1) {}
  QVector2D position;    // World-space position
  QVector3D color;       // RGB [0..1]
  float base_size;       // Base dot radius in world units
  float twinkle_phase;   // For subtle per-star twinkle offset
  QString title;
  QString artist;
  QString album_artist;
  QString album;
  QUrl art_url;          // URL for cover art
  QUrl url;              // Song URL
  int song_id;
  float bpm;
  int all_songs_index;   // Index into GalaxyMapView::all_songs_
  QString genre;
  QString album_key;
};

struct GalaxyConstellation {
  QVector<int> star_indices;
  QVector<QPair<int, int>> edges;
  QVector2D centroid; // in world coordinates
  QString label;
};

class GalaxyMapView : public QWidget {
  Q_OBJECT

 public:
  explicit GalaxyMapView(Application *app, QWidget *parent = nullptr);
  ~GalaxyMapView() override;

  void Init();
  // Reset the scan flag
  void ResetScanFlag();

  // Zoom thresholds (world-space pixels per unit)
  static constexpr float kZoomStarView      = 0.18f;   // < zoomed out
  static constexpr float kZoomConstellation = 3.5f;    // mid zoom
  // above kZoomConstellation => planet view

 protected:
  void paintEvent(QPaintEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseDoubleClickEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  void showEvent(QShowEvent *event) override;
  void leaveEvent(QEvent *event) override;

 private Q_SLOTS:
  void updateFrame();
  void onSongsChanged(const SongList &songs);
  void onAlbumArtLoaded(const QString &album_key, const QPixmap &pixmap);
  void onAlbumCoverLoaded(quint64 id, const AlbumCoverLoaderResult &result);

 private:
  void fetchSongsFromBackend(bool force_scan = false);
  void DeepEmbeddings(bool force_scan = false);
  void buildPlaceholderStars();
  void tryLoadFromModel();
  void buildStars(const SongList &songs, bool force_scan = false);
  void loadStarArt(int idx);
  void drawStarView(QPainter &p);
  void drawConstellationView(QPainter &p);
  void drawPlanetView(QPainter &p);
  void drawParallaxBackground(QPainter &p);

  QPointF worldToScreen(QVector2D world) const;
  QVector2D screenToWorld(QPointF screen) const;

  Application *app_;
  QTimer *timer_;

  float zoom_;       // screen pixels per world unit
  QVector2D pan_;    // camera center in world units
  QVector2D velocity_;
  bool is_dragging_;
  QPointF last_mouse_pos_;
  QPointF press_start_pos_;

  float anim_time_;  // accumulated animation time (seconds)

  QVector<GalaxyStar> stars_;
  QVector<GalaxyConstellation> constellations_;
  SongList all_songs_;  // authoritative song list from backend

  QMap<QString, QPixmap> album_art_cache_;       // album_key -> pixmap
  QMap<QString, QPixmap> album_art_thumb_cache_; // album_key -> 64x64 thumb
  QSet<QString> album_art_loading_;              // album keys currently being loaded
  QMap<quint64, QString> loader_id_to_album_key_;

  int hovered_star_;  // index of hovered star, -1 if none
  int selected_star_; // index of selected star, -1 if none
  float select_pulse_; // pulse ring animation [0..1]
  
  QFileSystemWatcher *db_watcher_;
  QProcess *deep_embedding_process_;
  QString deep_embedding_status_;
  float deep_embedding_progress_;
  QTimer *live_reveal_timer_;
  bool deep_embedding_started_;
  bool deep_embedding_scan_once_;
};

#endif // GALAXYMAPVIEW_H
