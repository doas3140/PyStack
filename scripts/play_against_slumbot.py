'''
original script: https://github.com/happypepper/DeepHoldem/blob/master/Source/Player/slumbot_player.py
'''

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
import os
import sys
import time

os.chdir('..')
sys.path.append( os.path.join(os.getcwd(),'src') )


def main():
    from Player.continual_resolving import ContinualResolving
    bot = ContinualResolving()
    name = 'pystackv1'
    password = 'pystackv1'
    play_against_slumbot(bot, name, password)


def play_against_slumbot(bot, slumbot_acc_name, slumbot_acc_password):

    slumbot_utils = SlumBotUtils()

    driver = webdriver.Chrome()
    driver.get("http://slumbot.com")

    time.sleep(2)
    driver.execute_script(slumbot_utils.response_fun)

    signup_button = driver.find_element_by_id("login_trigger")
    signup_button.click()
    time.sleep(1)
    name = driver.find_element_by_id("loginname")
    password = driver.find_element_by_id("loginpw")
    name.send_keys(slumbot_acc_name)
    time.sleep(0.5)
    password.send_keys(slumbot_acc_password)
    time.sleep(0.5)
    password.send_keys(Keys.RETURN)
    time.sleep(1)

    hand_no = 0

    while True:
        nexthand_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "nexthand")))
        time.sleep(1)
        nexthand_button.click()
        hand_no += 1

        player_is_small_blind = True
        while True:
            print(1)
            fold_button = driver.find_element_by_id("fold")
            nexthand_button = driver.find_element_by_id("nexthand")
            action_td = driver.find_element_by_id("currentaction")

            if fold_button.is_displayed() and fold_button.is_enabled():
                if action_td.text:
                    player_is_small_blind = False
                break
            if nexthand_button.is_displayed() and nexthand_button.is_enabled():
                break
            time.sleep(1)

        hand = driver.execute_script("return global_data[\"holes\"]")
        card1, card2 = hand[:2], hand[2:]
        # print('STARTING NEW HAND', hand, is_small_blind)
        bot.start_new_hand(card1, card2, player_is_small_blind)
        bot_bet = 50 if player_is_small_blind else 100

        # new hand
        while True:
            nexthand_button = driver.find_element_by_id("nexthand")
            if nexthand_button.is_displayed() and nexthand_button.is_enabled():
                break
            hole = driver.execute_script("return global_data[\"holes\"]")
            actions = driver.execute_script("return global_data[\"action\"]")
            board = driver.execute_script("return global_data[\"board\"]")

            fold_button = driver.find_element_by_id("fold")
            call_button = driver.find_element_by_id("call")
            check_button = driver.find_element_by_id("check")
            halfpot_button = driver.find_element_by_id("halfpot")
            pot_button = driver.find_element_by_id("pot")
            allin_button = driver.find_element_by_id("allin")

            while True:
                if (call_button.is_displayed() and call_button.is_enabled()) or (allin_button.is_displayed() and allin_button.is_enabled()):
                    break
                time.sleep(1)

            _, opponent_bet = slumbot_utils.acpcify_actions(actions)
            response = bot.compute_action(board_string=board, player_bet=bot_bet, opponent_bet=opponent_bet)
            advice, amount = response['action'], response['amount']
            if amount == 20000:
                advice = 'allin'

            if advice == "call":
                if call_button.is_displayed() and call_button.is_enabled():
                    call_button.click()
                    bot_bet = opponent_bet
                elif check_button.is_displayed() and check_button.is_enabled():
                    check_button.click()
            elif advice == "fold" and fold_button.is_displayed() and fold_button.is_enabled():
                fold_button.click()
            elif advice == "allin" and allin_button.is_displayed() and allin_button.is_enabled():
                allin_button.click()
                bot_bet = 20000
            elif advice == 'raise' and opponent_bet * 2 + opponent_bet > 20000:
                if check_button.is_displayed() and check_button.is_enabled():
                    check_button.click()
                elif call_button.is_displayed() and call_button.is_enabled():
                    call_button.click()
                else:
                    print('ERROR: folding')
                    fold_button.click()
                bot_bet = 20000
            elif advice == 'raise' and pot_button.is_displayed() and pot_button.is_enabled():
                pot_button.click()
                if bot_bet == 50 and opponent_bet == 100:
                    bot_bet = 300
                else:
                    bot_bet = opponent_bet * 2 + opponent_bet

            else:
                print('ERROR: folding')
                fold_button.click()

            while True:
                time.sleep(1)
                if (call_button.is_displayed() and call_button.is_enabled()) or (allin_button.is_displayed() and allin_button.is_enabled()):
                    break
                if nexthand_button.is_displayed() and nexthand_button.is_enabled():
                    break

class SlumBotUtils:
    def __init__(self):
        self.response_fun = """
        response = function(data) {
            global_data = data
            // We increment actionindex even when we get an error message back.
            // If we didn't, then when we retried the action that triggered the error,
            // it would get flagged as a duplicate.
            ++actionindex;
            if ("errormsg" in data) {
        	var errormsg = data["errormsg"];
        	$("#msg").text(errormsg);
        	// Some errors end the hand (e.g., server timeout)
        	// Would it be cleaner to treat a server timeout like a client
        	// timeout?  Return msg rather than errormsg?
        	if ("hip" in data) {
        	    handinprogress = (data["hip"] === 1);
        	}
        	// Need this for server timeout.  Want to enable the "Next Hand"
        	// button and disable all the other buttons.
        	enableactions();
        	return;
            } else if ("msg" in data) {
        	var msg = data["msg"];
        	$("#msg").text(msg);

            } else {
        	$("#msg").text("");
            }

            if (actiontype === 1) {
        	addourcheck();
            } else if (actiontype === 2) {
        	addourcall();
            } else if (actiontype === 3) {
        	addourfold();
            } else if (actiontype === 4) {
        	addourbet();
            }
            $("#betsize").val("");
            potsize = data["ps"];
            ourbet = data["ourb"];
            oppbet = data["oppb"];
            var lastcurrentaction = currentaction;
            currentaction = data["action"];
            var actiondisplay = currentaction;
            $("#currentaction").text(actiondisplay);
            var overlap = currentaction.substring(0, lastcurrentaction.length);
            if (overlap !== lastcurrentaction) {
        	console.log("Overlap " + overlap);
        	console.log("Last current action " + lastcurrentaction);
            } else {
        	var newaction = currentaction.substring(lastcurrentaction.length,
        						currentaction.length);
        	oppactionmessage(newaction);
            }

            parsedata(data);
            drawall(aftershowdown);
            lifetimetotal = data["ltotal"];
            lifetimeconf = data["lconf"];
            lifetimebaselinetotal = data["lbtotal"];
            lifetimebaselineconf = data["lbconf"];
            numlifetimehands = data["lhands"];
            showdowntotal = data["sdtotal"];
            showdownconf = data["sdconf"];
            numshowdownhands = data["sdhands"];
            blbshowdowntotal = data["blbsdtotal"];
            blbshowdownconf = data["blbsdconf"];
            blbnumshowdownhands = data["blbsdhands"];
            clbshowdowntotal = data["clbsdtotal"];
            clbshowdownconf = data["clbsdconf"];
            clbnumshowdownhands = data["clbsdhands"];
            if (username !== "") displaystats();
            if (! handinprogress) {
        	sessiontotal = data["stotal"];
        	$("#sessiontotal").text(sessiontotal);
        	numsessionhands = data["shands"];
        	$("#numhands").text(numsessionhands);
        	var outcome = data["outcome"];
        	if (outcome > 0) {
        	    $("#outcome").text("You won a pot of " + outcome + "!");
        	} else if (outcome < 0) {
        	    $("#outcome").text("Slumbot won a pot of " + -outcome +
        			       "!");
        	} else {
                    $("#outcome").text("You chopped!");
        	}
            } else {
        	$("#outcome").text("");
        	starttimer();
            }
            enableactions();
        };
        """


    def acpcify_actions(self, actions):
        actions = actions.replace("b","r")
        actions = actions.replace("k","c")
        streets = actions.split("/")
        max_bet = 0
        for i, street_actions in enumerate(streets):
            bets = street_actions.split("r")
            max_street_bet = max_bet
            for j, betstr in enumerate(bets):
                try:
                    flag = False
                    if len(betstr) > 1 and betstr[-1] == 'c':
                        flag = True
                        betstr = betstr.replace("c","")
                    bet = int(betstr)
                    bet += max_bet
                    max_street_bet = max(max_street_bet, bet)
                    bets[j] = str(bet)
                    if flag:
                        bets[j] += "c"
                    bets[j] = "r" + bets[j]
                except ValueError:
                    continue
            max_bet = max_street_bet
            if max_bet == 0:
                max_bet = 100
            good_string = "".join(bets)
            streets[i] = good_string
        return "/".join(streets), max_bet





if __name__ == '__main__':
    main()




#
