--[[
  Copyright 2014 Google Inc. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
]]--

function pair_opr(hardness)
  local out1, out2 = get_operands(hardness, 2)
  local a, b = table.unpack(out1)
  local c, d = table.unpack(out2)
  if random(2) == 1 then
    eval = a.eval + b.eval
    expr = string.format("(%s+%s)", a.expr, b.expr)
    eval1 = c.eval + d.eval
    expr1 = string.format("(%s+%s)", c.expr, d.expr)
  else
    eval = a.eval - b.eval
    expr = string.format("(%s-%s)", a.expr, b.expr)
    eval1 = c.eval - d.eval
    expr1 = string.format("(%s-%s)", c.expr, d.expr)
  end
  return {}, {}, expr, eval, expr1, eval1
end

function smallmul_opr(hardness)
  local out1, out2 = get_operand(hardness)
  local expr, eval = table.unpack(out1)
  local expr1, eval1 = table.unpack(out2)
  local b = random(4 * hardness())
  local eval = eval * b
  local eval1 = eval1 * b
  if random(2) == 1 then
    expr = string.format("(%s*%d)", expr, b)
    expr1 = string.format("(%s*%d)", expr1, b)
  else
    expr = string.format("(%d*%s)", b, expr)
    expr1 = string.format("(%d*%s)", b, expr1)
  end
  return {}, {}, expr, eval, expr1, eval1
end

function equality_opr(hardness)
  local out1, out2 = get_operand(hardness)
  local expr, eval = table.unpack(out1)
  local expr1, eval1 = table.unpack(out2)
  return {}, {}, expr, eval, expr1, eval1
end

function vars_opr(hardness)
  local var = variablesManager:get_unused_variables(1)
  local out1, out2 = get_operands(hardness, 2)
  local a, b = table.unpack(out1)
  local c, d = table.unpack(out2)
  if random(2) == 1 then
    eval = a.eval + b.eval
    code = {string.format("%s=%s;", var, a.expr)}
    expr = string.format("(%s+%s)", var, b.expr)
    eval1 = c.eval + d.eval
    code1 = {string.format("%s=%s;", var, c.expr)}
    expr1 = string.format("(%s+%s)", var, d.expr)
  else
    eval = a.eval - b.eval
    code = {string.format("%s=%s;", var, a.expr)}
    expr = string.format("(%s-%s)", var, b.expr)
    eval1 = c.eval - d.eval
    code1 = {string.format("%s=%s;", var, c.expr)}
    expr1 = string.format("(%s-%s)", var, d.expr)
  end
  return code, code1, expr, eval, expr1, eval1
end

function small_loop_opr(hardness)
  local r_small = hardness()
  local var = variablesManager:get_unused_variables(1)
  local out1, out2 = get_operands(hardness, 2)
  local a, b = table.unpack(out1)
  local c, d = table.unpack(out2)
  local loop = random(4 * hardness())
  local op = ""
  local val = 0
  if random(2) == 2 then
    op = "+"
    eval = a.eval + loop * b.eval
    eval1 = c.eval + loop * d.eval
  else
    op = "-"
    eval = a.eval - loop * b.eval
    eval1 = c.eval - loop * d.eval
  end
  local code = {string.format("%s=%s", var, a.expr),
                string.format("for x in range(%d):%s%s=%s", loop, var,
                                                            op, b.expr)}
    local code1 = {string.format("%s=%s", var, c.expr),
                  string.format("for x=1,%d do %s=%s%s%s; end", loop, var, var, op, d.expr)}
  local expr = var
  local expr1 = var
  return code, code1, expr, eval, expr1, eval1
end

function ifstat_opr(hardness)
  local r_small = hardness()
  local out1, out2 = get_operands(hardness, 4)
  local a, b, c, d = table.unpack(out1)
  local p, q, r, s = table.unpack(out2)
  if random(2) == 1 then
    name = ">"
    if a.eval > b.eval then
      output = c.eval
    else
      output = d.eval
    end
  else
    name = "<"
    if a.eval < b.eval then
      output = c.eval
    else
      output = d.eval
    end
  end
  local expr = string.format("(%s if %s%s%s else %s)",
                             c.expr, a.expr, name, b.expr, d.expr)
    local expr1 = string.format("(if %s%s%s then %s else %s end)",
                               a.expr, name, b.expr, c.expr, d.expr)
  return {}, {}, expr, output, expr1, output
end
