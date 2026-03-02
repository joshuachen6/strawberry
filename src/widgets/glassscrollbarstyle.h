#pragma once

#include <QProxyStyle>
#include <QPainter>
#include <QStyleOption>
#include <QWidget>
#include <QScrollBar>
#include <QApplication>

class GlassScrollBarStyle : public QProxyStyle {
  Q_OBJECT

 public:
  // Uses the current app style as the base (by name, avoids ownership issues)
  explicit GlassScrollBarStyle() : QProxyStyle(QApplication::style()->objectName()) {}

  void drawComplexControl(QStyle::ComplexControl control, const QStyleOptionComplex *option, QPainter *painter, const QWidget *widget = nullptr) const override {
    if (control != QStyle::CC_ScrollBar) {
      QProxyStyle::drawComplexControl(control, option, painter, widget);
      return;
    }

    const QStyleOptionSlider *scrollbar = qstyleoption_cast<const QStyleOptionSlider *>(option);
    if (!scrollbar) return;

    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, true);

    // Clear to transparent first
    painter->setCompositionMode(QPainter::CompositionMode_Source);
    painter->fillRect(option->rect, Qt::transparent);
    painter->setCompositionMode(QPainter::CompositionMode_SourceOver);

    const bool horizontal = scrollbar->orientation == Qt::Horizontal;
    const QRect groove = subControlRect(control, option, QStyle::SC_ScrollBarGroove, widget);
    const QRect handle = subControlRect(control, option, QStyle::SC_ScrollBarSlider, widget);

    // Subtle dark track
    const int radius = horizontal ? groove.height() / 2 : groove.width() / 2;
    painter->setPen(Qt::NoPen);
    painter->setBrush(QColor(0, 0, 0, 30));
    painter->drawRoundedRect(groove, radius, radius);

    // White rounded handle
    const bool hovered = scrollbar->activeSubControls & QStyle::SC_ScrollBarSlider;
    painter->setBrush(QColor(255, 255, 255, hovered ? 150 : 90));
    painter->drawRoundedRect(handle.adjusted(1, 1, -1, -1), radius - 1, radius - 1);

    painter->restore();
  }

  int pixelMetric(QStyle::PixelMetric metric, const QStyleOption *option = nullptr, const QWidget *widget = nullptr) const override {
    if (metric == QStyle::PM_ScrollBarExtent) return 8;
    if (metric == QStyle::PM_ScrollBarSliderMin) return 30;
    return QProxyStyle::pixelMetric(metric, option, widget);
  }
};
