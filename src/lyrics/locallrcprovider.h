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

#ifndef LOCALLRCPROVIDER_H
#define LOCALLRCPROVIDER_H

#include "lyrics/lyricsprovider.h"

class LocalLRCProvider : public LyricsProvider {
  Q_OBJECT

 public:
  explicit LocalLRCProvider(QObject *parent = nullptr);

 protected Q_SLOTS:
  void StartSearch(const int id, const LyricsSearchRequest &request) override;
};

#endif  // LOCALLRCPROVIDER_H
