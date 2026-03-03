/*
 * Strawberry Music Player
 * This file was part of Clementine.
 * Copyright 2010, David Sansome <me@davidsansome.com>
 *
 * Strawberry is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Strawberry is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Strawberry.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <QWidget>
#include <QSlider>
#include <QPoint>
#include <QRect>
#include <QStyle>
#include <QStyleOption>
#include <QEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QEnterEvent>
#include <QTimer>
#include <QStylePainter>
#include <QPainterPath>
#include <cmath>

#include "utilities/timeutils.h"
#include "constants/timeconstants.h"
#ifndef Q_OS_MACOS
#  include "tracksliderpopup.h"
#endif
#include "tracksliderslider.h"

TrackSliderSlider::TrackSliderSlider(QWidget *parent)
    : QSlider(parent),
#ifndef Q_OS_MACOS
      popup_(new TrackSliderPopup(window())),
#endif
      mouse_hover_seconds_(0),
      wheel_accumulator_(0),
      fluid_timer_(new QTimer(this)),
      fluid_offset_(0.0),
      is_playing_(false) {

  setMouseTracking(true);
#ifndef Q_OS_MACOS
  popup_->hide();
  QObject::connect(this, &TrackSliderSlider::valueChanged, this, &TrackSliderSlider::UpdateDeltaTime);
#endif

  QObject::connect(fluid_timer_, &QTimer::timeout, this, [this]() {
    if (maximum() > minimum() && is_playing_) {
      fluid_offset_ -= 0.08;
      update();
    }
  });
  fluid_timer_->start(30);

}

void TrackSliderSlider::mousePressEvent(QMouseEvent *e) {
  // QSlider asks QStyle which mouse button should do what (absolute move or page step).
  // We force our own behaviour here because it makes more sense for a music player IMO.

  Qt::MouseButton new_button = e->button();
  if (e->button() == Qt::LeftButton) {
    int abs_buttons = style()->styleHint(QStyle::SH_Slider_AbsoluteSetButtons);
    if (abs_buttons & Qt::LeftButton) {
      new_button = Qt::LeftButton;
    }
    else if (abs_buttons & Qt::MiddleButton) {
      new_button = Qt::MiddleButton;
    }
    else if (abs_buttons & Qt::RightButton) {
      new_button = Qt::RightButton;
    }
  }

  QMouseEvent new_event(e->type(), e->pos(), e->globalPosition(), new_button, new_button, e->modifiers());
  QSlider::mousePressEvent(&new_event);

  if (new_event.isAccepted()) {
    e->accept();
  }

}

void TrackSliderSlider::mouseReleaseEvent(QMouseEvent *e) {

  QSlider::mouseReleaseEvent(e);
  if (e->button() == Qt::XButton1) {
    Q_EMIT Previous();
  }
  else if (e->button() == Qt::XButton2) {
    Q_EMIT Next();
  }
  e->accept();

}

void TrackSliderSlider::mouseMoveEvent(QMouseEvent *e) {

  QSlider::mouseMoveEvent(e);

  // Borrowed from QSliderPrivate::pixelPosToRangeValue
  QStyleOptionSlider opt;
  initStyleOption(&opt);
  QRect gr = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderGroove, this);
  QRect sr = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, this);

  int slider_length = sr.width();
  int slider_min = gr.x();
  int slider_max = gr.right() - slider_length + 1;

  mouse_hover_seconds_ = QStyle::sliderValueFromPosition(minimum() / static_cast<int>(kMsecPerSec), maximum() / static_cast<int>(kMsecPerSec), e->pos().x() - slider_length / 2 - slider_min + 1, slider_max - slider_min);

#ifndef Q_OS_MACOS
  popup_->SetText(Utilities::PrettyTime(mouse_hover_seconds_));
  UpdateDeltaTime();
  popup_->SetPopupPosition(mapTo(window(), QPoint(e->pos().x(), rect().center().y())));
#endif

}

void TrackSliderSlider::wheelEvent(QWheelEvent *e) {

  const int scroll_state = wheel_accumulator_ + e->angleDelta().y();
  const int steps = scroll_state / WHEEL_ROTATION_TO_SEEK;
  wheel_accumulator_ = scroll_state % WHEEL_ROTATION_TO_SEEK;

  if (steps < 0) {
    Q_EMIT SeekBackward();
  }
  else if (steps > 0) {
    Q_EMIT SeekForward();
  }
  e->accept();

}

void TrackSliderSlider::enterEvent(QEnterEvent *e) {

  QSlider::enterEvent(e);
#ifndef Q_OS_MACOS
  if (isEnabled()) {
    popup_->show();
  }
#endif

}

void TrackSliderSlider::leaveEvent(QEvent *e) {

  QSlider::leaveEvent(e);
#ifndef Q_OS_MACOS
  if (popup_->isVisible()) {
    popup_->hide();
  }
#endif

}

void TrackSliderSlider::keyPressEvent(QKeyEvent *event) {

  if (event->key() == Qt::Key_Left || event->key() == Qt::Key_Down) {
    Q_EMIT SeekBackward();
    event->accept();
  }
  else if (event->key() == Qt::Key_Right || event->key() == Qt::Key_Up) {
    Q_EMIT SeekForward();
    event->accept();
  }
  else {
    QSlider::keyPressEvent(event);
  }

}

void TrackSliderSlider::UpdateDeltaTime() {

#ifndef Q_OS_MACOS
  if (popup_->isVisible()) {
    int delta_seconds = mouse_hover_seconds_ - (value() / static_cast<int>(kMsecPerSec));
    popup_->SetSmallText(Utilities::PrettyTimeDelta(delta_seconds));
  }
#endif

}

void TrackSliderSlider::paintEvent(QPaintEvent *e) {
  Q_UNUSED(e);

  // We draw the standard groove and handle, but we intermingle our fluid wave!
  QStylePainter p(this);
  QStyleOptionSlider opt;
  initStyleOption(&opt);

  // We want to detect if MoodbarProxyStyle is handling it. If so, and if moodbar is active,
  // MoodbarProxyStyle::drawComplexControl will draw everything regardless of subControls.
  // Actually, we can check if we have a proxy style that intercepts.
  // But wait, it's safer to just do the normal Qt painting which works with standard styles.

  opt.subControls = QStyle::SC_SliderGroove;
  p.drawComplexControl(QStyle::CC_Slider, opt);

  if (opt.maximum > opt.minimum) {
    QRect grooveRect = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderGroove, this);
    QRect handleRect = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, this);

    int playhead_x = handleRect.center().x();
    QRect playedRect = grooveRect;
    playedRect.setRight(playhead_x);

    // Minor adjustment to look better inside the groove
    playedRect.adjust(2, 2, 0, -2);

    if (playedRect.width() > 0) {
      p.save();
      p.setClipRect(playedRect);
      p.setRenderHint(QPainter::Antialiasing);

      QColor color1 = palette().color(QPalette::Highlight);
      color1.setAlpha(120);
      QColor color2 = palette().color(QPalette::Highlight);
      color2.setAlpha(180);

      int width = playedRect.width();
      int height = playedRect.height();
      double amp = height / 3.0;

      auto drawWave = [&](double phase1, double phase2, double phase3, QColor c, double baseFreq) {
        QPainterPath path;
        path.moveTo(playedRect.left(), playedRect.bottom());
        for (int x = 0; x <= width; x += 2) {
          // Complex ripple from multiple offset sine waves
          double wave1 = std::sin(baseFreq * x + phase1);
          double wave2 = std::sin(baseFreq * 1.618 * x + phase2) * 0.5;
          double wave3 = std::sin(baseFreq * 0.618 * x + phase3) * 0.3;
          double combined = (wave1 + wave2 + wave3) / 1.8;
          double y = playedRect.center().y() + amp * combined;
          path.lineTo(playedRect.left() + x, y);
        }
        path.lineTo(playedRect.right(), playedRect.bottom());
        path.closeSubpath();
        p.setPen(Qt::NoPen);
        p.setBrush(c);
        p.drawPath(path);
      };

      double freq = 2 * M_PI / 80.0;
      drawWave(fluid_offset_, fluid_offset_ * 1.3, fluid_offset_ * 0.7, color1, freq);
      drawWave(fluid_offset_ * 1.1 + M_PI, fluid_offset_ * 1.5 + M_PI/2, fluid_offset_ * 0.8 + M_PI, color2, freq * 1.1);

      p.restore();
    }
  }

  opt.subControls = QStyle::SC_SliderHandle;
  p.drawComplexControl(QStyle::CC_Slider, opt);

}
