# Hardware: what to buy, how to set it up, the first flight

The brain flies a stock **Bitcraze Crazyflie 2.1+**. Fly it in SITL first (`docs/phase2-deploy.md`).

## What to buy

Prices change. Check the [Bitcraze store](https://store.bitcraze.io) or a reseller like
RobotShop (US, about $240 for the 2.1+) before you order.
Prices below: Bitcraze store, seen 2026-10-09 (US dollars).

### Must have

| Item | Why | Price |
|---|---|---|
| Crazyflie 2.1+ | the drone (27–29 g) | $240 |
| Crazyradio 2.0 | USB radio: the PC talks to the drone through it | $43 |
| A positioning deck (pick one below) | the drone must know where it is, or the brain flies blind | $55–$95 |
| Spare propellers (4-packs, 2 CW + 2 CCW) | you will break some | $3 each pack |
| Spare 250 mAh batteries + charger | one battery is about 7 minutes of flying | $6.50, or $10.50 with a USB charger |

The "Getting started bundle – Crazyflie 2.1+" ($270–285) has the drone and the radio together.

### Positioning: two choices

| Choice | Cost | Good | Not so good |
|---|---|---|---|
| **Flow deck v2** (the cheap start) | $55 | clip on and fly; height from a laser, sideways from a camera looking down | position drifts a few cm per minute; needs a textured floor (not plain white, not glossy); height up to about 4 m |
| **Lighthouse deck + 2 base stations** (precise) | deck $95, each Lighthouse V2 base station $240; "Lighthouse explorer bundle – Crazyflie 2.1+" $810–858 | about 1 cm, no drift, works anywhere in the room | base stations must be mounted high in corners and calibrated |

The project brief also quoted about **$1,074** for a Lighthouse Explorer Bundle. That matches Bitcraze's
*Brushless* version ($1,050–1,098). For our brushed 2.1+ the 2.1+ bundle is enough. Check what each bundle contains.

**Start with the Flow deck.** Move to Lighthouse when the waypoint brain needs centimetre accuracy.

### Nice to have

| Item | Why |
|---|---|
| A net, or a soft mat and a corner with nothing breakable | crashes will happen |
| Spare motors and motor mounts ($6.50 for 4 mounts) | a hard crash can break a mount |

## Software setup (once)

| Step | How |
|---|---|
| 1. Python packages | `pip install -e .[deploy]` (cflib, onnx, onnxruntime at the versions we tested) |
| 2. Radio driver (Windows) | plug in the Crazyradio, run [Zadig](https://zadig.akeo.ie), pick "Crazyradio", install the **libusbK** driver. See Bitcraze's [USB permissions guide](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/) |
| 2. Radio driver (Linux) | add the udev rules from the same guide, then unplug and replug |
| 3. cfclient | `pip install cfclient`, then `cfclient`. Bitcraze's PC app: connect, watch the battery, read logs ([user guide](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/userguides/userguide_client/)) |
| 4. Update the firmware | in cfclient: Connect -> Bootloader -> pick the latest release -> Program. Do the Crazyradio too ([guide](https://www.bitcraze.io/documentation/repository/crazyradio2-firmware/main/)) |
| 5. Find the radio address | cfclient -> Scan. A new drone is `radio://0/80/2M/E7E7E7E7E7` |
| 6. Check the deck | cfclient -> Flight tab: the position and height should change when you lift the drone by hand |

Bitcraze's own start guide: [Getting started with the Crazyflie 2.1+](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/).

## First flight checklist

Fly the **real-world brains** (`policies/real_hover.pt`, `policies/real_waypoint.pt`). They were trained on noisy,
late sensors and randomized drones (docs/phase2-sensors.md). The `*_attitude.pt` brains were trained on perfect
data: on realistic drones the waypoint one crashed up to 16 % of the time.

Do these in order. Do not skip one.

| # | Check |
|---|---|
| 1 | The same brain and mission already flew in SITL: `python -m mindflight.deploy --policy policies/real_hover.pt --sitl` |
| 2 | Props on the right motors (the arrows on the arms), pushed fully down |
| 3 | Battery charged (4.1 V or more in cfclient), strapped down, deck clicked in |
| 4 | Open space: 2 m clear on every side, nothing above. People behind you. Soft floor or a net |
| 5 | Textured floor (Flow deck) or Lighthouse calibrated |
| 6 | Drone on the floor, level, nose pointing along +x (away from you) |
| 7 | Measure the hover command: `python -m mindflight.deploy --policy policies/real_hover.pt --uri --calibrate-hover --height 0.3` |
| 8 | First real flight: a **0.3 m hover**, nothing else: `python -m mindflight.deploy --policy policies/real_hover.pt --uri --height 0.3 --hover-command <number from step 7>` |
| 9 | Hand on the keyboard the whole time. **k = motors off** (Ctrl+C works too) |
| 10 | Read the flight log (`runs/flights/`) before trying waypoints |

Then waypoints, close together and low first: `--waypoints 0.3,0,0.4 0.3,0.3,0.4 0,0,0.4`.

### What stops the motors by itself

| What | Who |
|---|---|
| k / Ctrl+C | our runner, at once |
| Tilt past 55°, NaN, more than 0.5 m outside the fence | our runner |
| Radio silent for 0.5 s | the drone levels itself and keeps its thrust |
| Radio silent for 2 s | the drone stops its motors and **locks** until you restart it (power button) |
| Upside down, or tilted past 60° for 1 s | the drone ("crashed") |

Leaving the fence or a low battery (below 3.2 V for 1 s) makes our runner land, not drop.
