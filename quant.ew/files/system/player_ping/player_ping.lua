local ctx = dofile_once("mods/quant.ew/files/core/ctx.lua")
local net = dofile_once("mods/quant.ew/files/core/net.lua")
local rpc = net.new_rpc_namespace()

local gui = GuiCreate()

local module = {}

local pings = {}

-- "Borrowed" from MK VIII QF 2-puntaa NAVAL-ASE in Noita discord server.
-- https://discord.com/channels/453998283174576133/632303734877192192/1178002118368559175
local function world2gui( x, y )
    in_camera_ref = in_camera_ref or false

    local gui_n = GuiCreate()
    GuiStartFrame(gui_n)
    local w, h = GuiGetScreenDimensions(gui_n)
    GuiDestroy(gui_n)

    local vres_scaling_factor = w/( MagicNumbersGetValue( "VIRTUAL_RESOLUTION_X" ) + MagicNumbersGetValue( "VIRTUAL_RESOLUTION_OFFSET_X" ))
    local cam_x, cam_y = GameGetCameraPos()
    x, y = w/2 + vres_scaling_factor*( x - cam_x ), h/2 + vres_scaling_factor*( y - cam_y )

    return x, y, vres_scaling_factor
end

function module.on_world_update()
    GuiStartFrame(gui)

    GuiZSet(gui, 11)

    local ccx, ccy = GameGetCameraPos()
    local csx, csy, tcw, tch = GameGetCameraBounds()

    local cw = tcw - 10
    local ch = tch - 10
    local half_cw = cw / 2
    local half_ch = ch / 2

    local gui_id = 2

    local rebind = tonumber(ModSettingGet("quant.ew.rebind_ping"))
    if (not EntityHasTag(ctx.my_player.entity, "polymorphed") or EntityHasTag(ctx.my_player.entity, "ew_notplayer"))
            and (
            ((rebind == 42 and InputIsMouseButtonJustDown(3))
                    or (rebind ~= 42 and InputIsKeyJustDown(rebind)))
                        or InputIsJoystickButtonJustDown(0, 18)) then
        local x,y
        if GameGetIsGamepadConnected() then
            local tx, ty
            if ctx.spectating_over_peer_id == nil or ctx.spectating_over_peer_id == ctx.my_id then
                tx, ty = EntityGetTransform(ctx.my_player.entity)
            else
                tx, ty = EntityGetTransform(ctx.players[ctx.spectating_over_peer_id].entity)
            end
            x, y = InputGetJoystickAnalogStick(0, 1)
            x, y = x * 60 + tx, y * 60 + ty
        else
            x,y = DEBUG_GetMouseWorld()
        end
        rpc.send_ping(x, y)
    end

    local i = 1
    while i <= #pings do
        local pos = pings[i]
        local frame = pos[3]
        local peer_id = pos[4]
        if frame + 300 < GameGetFrameNum() then
            table.remove(pings, i)
            goto continue
        end

        local px = pos[1]
        local py = pos[2]
        local player_dir_x = px - ccx
        local player_dir_y = py - ccy
        local dist_sq = player_dir_x * player_dir_x + player_dir_y * player_dir_y
        -- local dist_sq = player_dir_x * player_dir_x + player_dir_y * player_dir_y
        -- player_dir_x = player_dir_x / dist
        -- player_dir_y = player_dir_y / dist

        local outside = false

        -- Contain the arrow in screen rect.
        if player_dir_x > half_cw then
            player_dir_y = player_dir_y / (player_dir_x / half_cw)
            player_dir_x = half_cw
            outside = true
        end
        if player_dir_x < -half_cw then
            player_dir_y = player_dir_y / (player_dir_x / -half_cw)
            player_dir_x = -half_cw
            outside = true
        end
        if player_dir_y > half_ch then
            player_dir_x = player_dir_x / (player_dir_y / half_ch)
            player_dir_y = half_ch
            outside = true
        end
        if player_dir_y < -half_ch then
            player_dir_x = player_dir_x / (player_dir_y / -half_ch)
            player_dir_y = -half_ch
            outside = true
        end

        local img_path = "mods/quant.ew/files/system/player/tmp/".. peer_id .."_ping.png"
        if outside then
            local scale = math.max(1 / 6, 0.75 - math.atan((math.sqrt(dist_sq) - tch) / 1280) / math.pi)
            local x, y = world2gui(ccx+player_dir_x, ccy+player_dir_y)
            GuiImage(gui, gui_id, x, y, img_path, 1, scale, 0, math.atan2(player_dir_y, player_dir_x) + math.pi/2)
        else
            local x, y = world2gui(pos[1], pos[2])
            GuiImage(gui, gui_id, x, y, img_path, 1, 0.75, 0, math.pi)
        end
        gui_id = gui_id + 1
        i = i + 1
        ::continue::
    end
 end

rpc.opts_everywhere()
function rpc.send_ping(x, y)
    table.insert(pings, {x, y, GameGetFrameNum(), ctx.rpc_peer_id})
end

return module