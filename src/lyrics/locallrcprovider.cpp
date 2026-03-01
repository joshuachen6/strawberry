/*
 * Strawberry Music Player
 * Copyright 2026, Gemini
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

#include "locallrcprovider.h"
#include <QFile>
#include <QFileInfo>
#include <QTextStream>

using namespace Qt::Literals::StringLiterals;

LocalLRCProvider::LocalLRCProvider(QObject *parent)
    : LyricsProvider(u"local-lrc"_s, true, false, nullptr, parent) {}

void LocalLRCProvider::StartSearch(const int id, const LyricsSearchRequest &request) {
  const QUrl url = request.song.url();
  if (!url.isLocalFile()) {
    Q_EMIT SearchFinished(id);
    return;
  }

  QString path = url.toLocalFile();
  QFileInfo fileInfo(path);
  QString lrcPath = fileInfo.path() + QLatin1Char('/') + fileInfo.completeBaseName() + u".lrc"_s;

  if (QFile::exists(lrcPath)) {
    QFile lrcFile(lrcPath);
    if (lrcFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
      QTextStream in(&lrcFile);
      QString lyrics = in.readAll();
      lrcFile.close();

      LyricsSearchResult result(lyrics);
      result.provider = name();
      result.artist = request.song.artist();
      result.title = request.song.title();
      result.score = 10.0f; // Perfect score since it's a local file

      LyricsSearchResults results;
      results.append(result);
      Q_EMIT SearchFinished(id, results);
      return;
    }
  }

  Q_EMIT SearchFinished(id);
}
