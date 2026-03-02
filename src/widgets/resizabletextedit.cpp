/*
 * Strawberry Music Player
 * Copyright 2022, Jonas Kvinge <jonas@jkvinge.net>
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

#include <QTextEdit>
#include <QResizeEvent>
#include <QTextBlockFormat>
#include <QTextCursor>

#include "resizabletextedit.h"

ResizableTextEdit::ResizableTextEdit(QWidget *parent)
    : QTextEdit(parent) {

  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setWordWrapMode(QTextOption::WordWrap);
  setLineWrapMode(QTextEdit::WidgetWidth);
  document()->setDocumentMargin(0);
  setViewportMargins(0, 0, 0, 0);
  setContentsMargins(0, 0, 0, 0);
  
  // Set default text option and alignment
  QTextOption option = document()->defaultTextOption();
  option.setAlignment(Qt::AlignLeft);
  document()->setDefaultTextOption(option);

}

QSize ResizableTextEdit::sizeHint() const {

  return QSize(std::max(QTextEdit::sizeHint().width(), 10), std::max(document()->size().toSize().height(), 10));

}

void ResizableTextEdit::resizeEvent(QResizeEvent *e) {

  QTextEdit::resizeEvent(e);
  updateGeometry();

}

void ResizableTextEdit::SetText(const QString &text) {

  // Normalize input string: remove \r and trim each line to ensure no hidden spaces
  QString normalized = text;
  normalized.remove(u'\r');
  QStringList lines = normalized.split(u'\n');
  for (QString &line : lines) line = line.trimmed();
  normalized = lines.join(u'\n');
  
  text_ = normalized;

  // Use setText to preserve HTML/rich formatting from the source
  QTextEdit::setText(normalized);

  // Force zero margins on the document AGAIN after setText (as setText resets document)
  document()->setDocumentMargin(0);

  // Apply a uniform block format to EVERY block in the document to ensure uniformity
  QTextBlockFormat fmt;
  fmt.setTextIndent(0);
  fmt.setIndent(0);
  fmt.setLeftMargin(0);
  fmt.setRightMargin(0);
  fmt.setTopMargin(0);
  fmt.setBottomMargin(0);
  fmt.setAlignment(Qt::AlignLeft);
  if (line_spacing_ > 0) {
    fmt.setLineHeight(line_spacing_, QTextBlockFormat::LineDistanceHeight);
  }

  QTextCursor cursor(document());
  cursor.select(QTextCursor::Document);
  cursor.setBlockFormat(fmt);

  updateGeometry();

}

void ResizableTextEdit::SetLineSpacing(int spacing) {
  line_spacing_ = spacing;
  if (!text_.isEmpty()) {
    SetText(text_);
  }
}
