# Card images

The web frontend (`web/app.js`) shows a photo on every result card.

## Per-centre photos — `services/NN.jpg`

Each record in `data/services.json` carries an `image` field, e.g.
`"image": "services/07.jpg"`. `services/01.jpg` … `services/76.jpg` are one
distinct photo per centre, in the same order as the records in
`data/services.json`. `src/api.py` `_photo_url()` uses this first.

## Fallback photos — `<care_type>.jpg`

Used only if a record has no `image`:

| file | `care_type` |
|---|---|
| `day-care.jpg` | `day-care` |
| `home-help.jpg` | `home-help` |
| `short-stay.jpg` | `short-stay` |
| `night-respite.jpg` | `night-respite` |
| `weekend-respite.jpg` | `weekend-respite` |
| `default.jpg` | anything else |

The frontend also renders an emoji + gradient fallback if a file is missing,
so a gap here never breaks a card.

## Source & licence

All photos are from [Pexels](https://www.pexels.com) under the **Pexels
licence** — free to use, commercial use allowed, **no attribution required**
(<https://www.pexels.com/license/>). Cropped to ~16:9. To change one, drop a
replacement in at the same path/filename — no code change needed.
