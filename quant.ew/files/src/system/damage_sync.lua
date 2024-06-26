-- This module allows to handle getting damaged locally and redirects that damage to host.

local ctx = dofile_once("mods/quant.ew/files/src/ctx.lua")
local net = dofile_once("mods/quant.ew/files/src/net.lua")
local np = require("noitapatcher")

local rpc = net.new_rpc_namespace()

local module = {}

module.recent_damage = 0
module.recent_message = "unknown"

np.CrossCallAdd("ew_ds_damaged", function (damage, message)
    module.recent_damage = module.recent_damage + damage
    module.recent_message = message
end)

function module.on_local_player_spawn(my_player)
    if not ctx.is_host then
        EntityAddComponent2(my_player.entity, "LuaComponent", {script_damage_received = "mods/quant.ew/files/cbs/send_damage_to_host.lua"})

        local damage_model = EntityGetFirstComponentIncludingDisabled(my_player.entity, "DamageModelComponent")
        -- ComponentSetValue2(damage_model, "damage_multipliers", "melee", 0)
    end
end

function module.on_world_update_client()
    if module.recent_damage ~= 0 and GameGetFrameNum() % 15 == 2 then
        rpc.deal_damage(module.recent_damage, module.recent_message)
        module.recent_damage = 0
        module.recent_message = "unknown"
    end
end

rpc.opts_reliable()
function rpc.deal_damage(damage, message)
    local message = GameTextGetTranslatedOrNot(message) .. " from "..ctx.rpc_player_data.name
    if ctx.is_host then
        -- local entity_thats_responsible = ctx.rpc_player_data.entity
        local host_entity_id = ctx.my_player.entity
        local protection_component_id = GameGetGameEffect(host_entity_id, "PROTECTION_ALL")
        if protection_component_id ~= 0 then
            EntitySetComponentIsEnabled(host_entity_id, protection_component_id, false)
        end
        EntityInflictDamage(host_entity_id, damage, "DAMAGE_CURSE", message, "NONE", 0, 0, nil)
        if protection_component_id ~= 0 then
            EntitySetComponentIsEnabled(host_entity_id, protection_component_id, true)
        end
    end
    GamePrint("Got ".. (damage*25) .." damage: "..message)
end

return module