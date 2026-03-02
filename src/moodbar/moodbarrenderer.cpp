/*
 * Strawberry Music Player
 * This file was part of Clementine.
 * Copyright 2012, David Sansome <me@davidsansome.com>
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

#include <QtGlobal>
#include <QObject>
#include <QArrayData>
#include <QList>
#include <QByteArray>
#include <QString>
#include <QImage>
#include <QPainter>
#include <QPalette>
#include <QColor>
#include <QRect>
#include <QSize>

#include "moodbarrenderer.h"
#include "constants/moodbarsettings.h"

ColorVector MoodbarRenderer::Colors(const QByteArray &data, const MoodbarSettings::Style style, const QPalette &palette) {

  const int samples = static_cast<int>(data.size() / 3);

  // Set some parameters based on the moodbar style
  StyleProperties properties;
  switch (style) {
    case MoodbarSettings::Style::Angry:
      properties = StyleProperties(samples / 360 * 9, 45, -45, 200, 100);
      break;
    case MoodbarSettings::Style::Frozen:
      properties = StyleProperties(samples / 360 * 1, 140, 160, 50, 100);
      break;
    case MoodbarSettings::Style::Happy:
      properties = StyleProperties(samples / 360 * 2, 0, 359, 150, 250);
      break;
    case MoodbarSettings::Style::Normal:
      properties = StyleProperties(samples / 360 * 3, 0, 359, 100, 100);
      break;
    case MoodbarSettings::Style::SystemPalette:
    default:{
      const QColor highlight_color(palette.color(QPalette::Active, QPalette::Highlight));

      properties.threshold_ = samples / 360 * 3;
      properties.range_start_ = (highlight_color.hsvHue() - 20 + 360) % 360;
      properties.range_delta_ = 20;
      properties.sat_ = highlight_color.hsvSaturation();
      properties.val_ = highlight_color.value() / 2;
    }
  }

  const unsigned char *data_p = reinterpret_cast<const unsigned char*>(data.constData());

  int hue_distribution[360];
  int total = 0;

  memset(hue_distribution, 0, sizeof(hue_distribution));

  ColorVector colors;
  colors.reserve(samples);
  // Read the colors, keeping track of some histograms
  for (int i = 0; i < samples; ++i) {
    QColor color;
    color.setRed(static_cast<int>(*data_p++));
    color.setGreen(static_cast<int>(*data_p++));
    color.setBlue(static_cast<int>(*data_p++));

    colors << color;

    const int hue = qMax(0, color.hue());
    if (hue_distribution[hue]++ == properties.threshold_) {
      total++;
    }
  }

  total = qMax(total, 1);

  // Remap the hue values to be between rangeStart and rangeStart + rangeDelta.
  // Every time we see an input hue above the threshold, increment the output hue by (1/total) * rangeDelta.
  for (int i = 0, n = 0; i < 360; i++) {
    hue_distribution[i] = ((hue_distribution[i] > properties.threshold_ ? n++ : n) * properties.range_delta_ / total + properties.range_start_) % 360;
  }

  // Now huedist is a hue mapper: huedist[h] is the new hue value for a bar with hue h
  for (ColorVector::iterator it = colors.begin(); it != colors.end(); ++it) {
    const int hue = qMax(0, it->hue());

    *it = QColor::fromHsv(qBound(0, hue_distribution[hue], 359), qBound(0, it->saturation() * properties.sat_ / 100, 255), qBound(0, it->value() * properties.val_ / 100, 255));
  }

  return colors;

}

void MoodbarRenderer::Render(const ColorVector &colors, QPainter *p, const QRect rect) {

  // Sample the colors and map them to screen pixels.
  ColorVector screen_colors;
  screen_colors.reserve(rect.width());
  for (int x = 0; x < rect.width(); ++x) {
    int r = 0;
    int g = 0;
    int b = 0;

    int start = static_cast<int>(x * colors.size() / rect.width());
    int end = static_cast<int>((x + 1) * colors.size() / rect.width());

    if (start == end) end = qMin(start + 1, static_cast<int>(colors.size() - 1));

    for (int j = start; j < end; j++) {
      r += colors[j].red();
      g += colors[j].green();
      b += colors[j].blue();
    }

    const int n = qMax(1, end - start);
    screen_colors.append(QColor(r / n, g / n, b / n));
  }

  // Find maximum brightness to normalize height for the Energy Silhouette
  int max_v = 1;
  for (const QColor &c : screen_colors) {
    int h, s, v;
    c.getHsv(&h, &s, &v);
    if (v > max_v) max_v = v;
  }

  // Draw the actual moodbar (Energy Silhouette).
  for (int x = 0; x < rect.width(); x++) {
    int h = 0, s = 0, v = 0;
    screen_colors[x].getHsv(&h, &s, &v);

    float energy_ratio = static_cast<float>(v) / static_cast<float>(max_v);
    // Smooth, non-linear scaling for a more dramatic silhouette
    energy_ratio = energy_ratio * energy_ratio * 0.3f + energy_ratio * 0.7f;
    float min_ratio = 0.15f;
    energy_ratio = qMax(min_ratio, energy_ratio);

    int col_height = static_cast<int>(static_cast<float>(rect.height()) * energy_ratio);
    if (col_height <= 0) col_height = 1;

    int start_y = rect.top() + rect.height() - col_height;
    int end_y = rect.bottom();

    for (int y = start_y; y <= end_y; ++y) {
      float y_ratio = 1.0f - static_cast<float>(y - start_y) / static_cast<float>(qMax(1, col_height - 1));

      // Make the top of the mountain brighter, and the base deeper/more saturated
      int current_v = qBound(0, static_cast<int>(static_cast<float>(v) * (0.4f + 0.6f * y_ratio)), 255);
      int current_s = qBound(0, static_cast<int>(static_cast<float>(s) * (1.0f - 0.4f * y_ratio)), 255);

      p->setPen(QColor::fromHsv(h, current_s, current_v));
      p->drawPoint(rect.left() + x, y);
    }
  }

}

QImage MoodbarRenderer::RenderToImage(const ColorVector &colors, const QSize size) {

  QImage image(size, QImage::Format_ARGB32_Premultiplied);
  QPainter p(&image);
  Render(colors, &p, image.rect());
  p.end();
  return image;

}

QString MoodbarRenderer::StyleName(const MoodbarSettings::Style style) {

  switch (style) {
    case MoodbarSettings::Style::Normal:
      return QObject::tr("Normal");
    case MoodbarSettings::Style::Angry:
      return QObject::tr("Angry");
    case MoodbarSettings::Style::Frozen:
      return QObject::tr("Frozen");
    case MoodbarSettings::Style::Happy:
      return QObject::tr("Happy");
    case MoodbarSettings::Style::SystemPalette:
      return QObject::tr("System colors");

    default:
      return QString();
  }

}
