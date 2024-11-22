local ctx = dofile_once("mods/quant.ew/files/core/ctx.lua")
local player_fns = dofile_once("mods/quant.ew/files/core/player_fns.lua")
local rpc = net.new_rpc_namespace()

local tether_length = ctx.proxy_opt.tether_length
local tether_length_2 = tether_length + 128

local module = {}

local ignore_tower = false

function is_in_box(x1, x2, y1, y2, x, y)
    return x1 < x and x < x2 and y1 < y and y < y2
end

function not_in_normal_area(x, y)
    return not (-5646 < x and x < 5120 and -1400 < y and y < 14336) and not is_in_box(5632, 7168, 14336, 15872, x, y)
end

function position_to_area_number(x, y)
    if np.GetGameModeNr() == 2 then
        if y < 1199 then
            return 1, 1199
        elseif y < 3759 then
            return 2, 3759
        elseif y < 6319 then
            return 3, 6319
        elseif y < 10415 then
            return 4, 10415
        elseif y < 12975 and (x < 2726 or x > 4135 or y < 12800) then
            return 5
        elseif is_in_box(5632, 7168, 14336, 15872, x, y) then
            return 10
        else
            return 6
        end
    elseif tonumber(SessionNumbersGetValue("NEW_GAME_PLUS_COUNT")) > 0 then
        if y < 1199 then
            return 1, 1199
        elseif y < 2735 then
            return 2, 2735
        elseif y < 6319 then
            return 3, 6319
        elseif y < 10415 then
            return 4, 10415
        elseif y < 12975 and (x < 2726 or x > 4135 or y < 12800) then
            return 5
        elseif is_in_box(5632, 7168, 14336, 15872, x, y) then
            return 10
        else
            return 6
        end
    else
        if y < 1199 then
            return 1, 1199
        elseif y < 2735 then
            return 2, 2735
        elseif y < 4783 then
            return 3, 4783
        elseif y < 6319 then
            return 4, 6319
        elseif y < 8367 then
            return 5, 8367
        elseif y < 10415 then
            return 6, 10415
        elseif y < 12975 and (x < 2726 or x > 4135 or y < 12800) then
            return 7
        elseif is_in_box(5632, 7168, 14336, 15872, x, y) then
            return 10
        else
            return 8
        end
    end
end

local function new_pos(given)
    if given ~= nil then
        return -678, given + 149
    else
        return 1914, 13119
    end
end

local function in_normal_hm(list, x, y)
    local x1 = -1024
    local x2 = 512
    local dy = 338
    for _, y1 in ipairs(list) do
        if is_in_box(x1, x2, y1, y1 + dy, x, y) then
            return true
        end
    end
    return false
end

local function not_in_hm(x, y)
    if np.GetGameModeNr() == 2 then
        local list = {1198, 3758, 6318, 10414}
        local in_hm = in_normal_hm(list, x, y)
        return not in_hm, not (in_hm or is_in_box(1536, 2726, 12798, 13312, x, y))
                or is_in_box(5632, 7168, 14336, 15872, x, y) --final room
    elseif tonumber(SessionNumbersGetValue("NEW_GAME_PLUS_COUNT")) > 0 then
        local list = {1198, 2734, 6318, 10414}
        local in_hm = in_normal_hm(list, x, y)
        return not in_hm, not (in_hm or is_in_box(1536, 2726, 12798, 13312, x, y))
                or is_in_box(5632, 7168, 14336, 15872, x, y) --final room
    else
        local list = {1198, 2734, 4782, 6318, 8366, 10414}
        local in_hm = in_normal_hm(list, x, y)
        return not in_hm, not (in_hm
                or is_in_box(1536, 2726, 12798, 13312, x, y) --last holy mountain
                or is_in_box(-4634, -4054, 2006, 2580, x, y) --meditation cube
                or is_in_box(-4060, -3656, 5078, 5660, x, y) --eye room
                or is_in_box(3578, 4080, 4048, 4640, x, y) --snow room
                or is_in_box(8700, 11300, 3550, 10240, x, y) --tower
                or is_in_box(5632, 7168, 14336, 15872, x, y) --final room
        )
    end
end

local no_tether = false

function module.on_client_spawned(peer_id, new_playerdata)
    local is_host = peer_id == ctx.host_id
    if is_host then
        local zone_ent = EntityLoad("mods/quant.ew/files/system/player_tether/zone_entity.xml")
        EntityAddChild(new_playerdata.entity, zone_ent)
        local particle_component = EntityGetFirstComponentIncludingDisabled(zone_ent, "ParticleEmitterComponent")
        ComponentSetValue2(particle_component, "area_circle_radius", tether_length, tether_length + 2)
        EntitySetComponentIsEnabled(zone_ent, particle_component, not no_tether)
    end
end

local function is_suitable_target(entity)
    return EntityGetIsAlive(entity) and not EntityHasTag(entity,"polymorphed")
end

local function tether_enable(to_enable, entity)
    local found = false
    for _, child in ipairs(EntityGetAllChildren(entity) or {}) do
        if EntityGetFilename(child) == "mods/quant.ew/files/system/player_tether/zone_entity.xml" then
            local emmiter = EntityGetFirstComponentIncludingDisabled(child, "ParticleEmitterComponent")
            EntitySetComponentIsEnabled(child, emmiter, to_enable)
            found = true
            break
        end
    end
    if not found then
        local child = EntityLoad("mods/quant.ew/files/system/player_tether/zone_entity.xml")
        EntityAddChild(entity, child)
        local emmiter = EntityGetFirstComponentIncludingDisabled(child, "ParticleEmitterComponent")
        EntitySetComponentIsEnabled(child, emmiter, to_enable)
    end
end

local function set_tether_length(length, entity)
    for _, child in ipairs(EntityGetAllChildren(entity) or {}) do
        if EntityGetFilename(child) == "mods/quant.ew/files/system/player_tether/zone_entity.xml" then
            local emmiter = EntityGetFirstComponentIncludingDisabled(child, "ParticleEmitterComponent")
            ComponentSetValue2(emmiter, "area_circle_radius", length, length + 2)
            break
        end
    end
end

local function float()
    local character_data = EntityGetFirstComponentIncludingDisabled(ctx.my_player.entity, "CharacterDataComponent")
    ComponentSetValue2(character_data, "mVelocity", 0, -80)
end

function rpc.teleport_to_tower()
    if ignore_tower then
        return
    end
    local x2, y2 = EntityGetTransform(ctx.my_player.entity)
    if is_in_box(9200, 11000, 8300, 9800, x2, y2) then
        return
    end
    async(function()
        EntitySetTransform(ctx.my_player.entity, 9740, 9100)
        wait(30)
        EntitySetTransform(ctx.my_player.entity, 9740, 9100)
        float()
    end)
end

local tether_length_3 = tether_length_2

local was_not_hm = false

local was_notplayer = false

function module.on_world_update()
    if GameGetFrameNum() % 10 == 7 then
        local host_playerdata = player_fns.peer_get_player_data(ctx.host_id, true)
        if ctx.proxy_opt.perma_death and (not ctx.my_player.status.is_alive or not host_playerdata.is_alive) then
            return
        end
        local x2, y2 = EntityGetTransform(ctx.my_player.entity)
        if is_in_box(9200, 11000, 4000, 8300, x2, y2) then
            ignore_tower = true
        end
        if np.GetGameModeNr() ~= 2
                and tonumber(SessionNumbersGetValue("NEW_GAME_PLUS_COUNT")) == 0
                and is_in_box(9200, 11000, 8300, 9800, x2, y2) then
            local any_not = false
            for _, player in pairs(ctx.players) do
                local x, y = EntityGetTransform(player.entity)
                if not is_in_box(9200, 11000, 4000, 9800, x, y) then
                    any_not = true
                end
                if is_in_box(0, 1000, -2000, 0, x, y) then
                    async(function()
                        EntitySetTransform(ctx.my_player.entity, 770, 900)
                        wait(30)
                        EntitySetTransform(ctx.my_player.entity, 770, 900)
                        float()
                    end)
                    return
                end
            end
            if any_not then
                rpc.teleport_to_tower()
            end
            return
        end
        if ctx.my_id == ctx.host_id then
            return
        end
        if host_playerdata == nil or host_playerdata.entity == nil or not EntityGetIsAlive(host_playerdata.entity) then
            return
        end
        local x1, y1 = host_playerdata.pos_x, host_playerdata.pos_y
        if x1 == nil or x2 == nil then
            return
        end
        if host_playerdata == nil
                or (not is_suitable_target(host_playerdata.entity)
                    and not (not not_in_normal_area(x1, y1) and not not_in_normal_area(x2, y2) and position_to_area_number(x1, y1) > position_to_area_number(x2, y2)))
                or not is_suitable_target(ctx.my_player.entity) then
            if host_playerdata ~= nil and host_playerdata.entity ~= nil and EntityGetIsAlive(host_playerdata.entity) then
                no_tether = true
                tether_enable(false, host_playerdata.entity)
                was_notplayer = true
            end
            return
        end
        if GameHasFlagRun("ending_game_completed") then
            if not no_tether then
                tether_enable(false, host_playerdata.entity)
                no_tether = true
            end
            return
        end
        local dx = x1-x2
        local dy = y1-y2
        local dist_sq = dx*dx + dy*dy
        local not_actual_hm, not_hm = not_in_hm(x1, y1)
        local _, i_not_in_hm = not_in_hm(x2, y2)
        local host_mx, host_my = host_playerdata.mouse_x, host_playerdata.mouse_y
        local dxm, dym = host_mx - x2, host_my - y2
        if x1 ~= nil and x2 ~= nil and (not_hm or (not not_actual_hm and y1 < y2)) and i_not_in_hm
                and dxm * dxm + dym * dym > tether_length * tether_length / 2 then
            if no_tether then
                tether_enable(true, host_playerdata.entity)
                no_tether = false
                if not was_not_hm or was_notplayer then
                    tether_length_3 = math.max(math.sqrt(dist_sq) + 256, tether_length_2)
                    set_tether_length(tether_length_3 - 128, host_playerdata.entity)
                end
            end
            if dist_sq > tether_length_3 * tether_length_3 then
                local x, y = x1, y1
                local my_pos, given = position_to_area_number(x2, y2)
                if not not_in_normal_area(x, y) and not not_in_normal_area(x2, y2) and position_to_area_number(x, y) > my_pos then
                    x, y = new_pos(given)
                end
                async(function()
                    EntitySetTransform(ctx.my_player.entity, x, y)
                    wait(40)
                    EntitySetTransform(ctx.my_player.entity, x, y)
                    float()
                end)
            elseif tether_length_3 > tether_length_2 then
                tether_length_3 = math.max(math.min(tether_length_3, math.sqrt(dist_sq) + 256), tether_length_2)
                set_tether_length(tether_length_3 - 128, host_playerdata.entity)
            end
        else
            no_tether = true
            tether_enable(false, host_playerdata.entity)
        end
        was_not_hm = not_hm
        was_notplayer = false
    end
end

return module