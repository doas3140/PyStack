// setup starting stacks
let STACK = 20000
$("#player_chips").text(STACK)
$("#opponent_chips").text(STACK)

// setup sockets with server
let SERVER_ADDRESS = 'http://localhost:8000'
let socket = io.connect(SERVER_ADDRESS)

$('#buttons_and_slider').hide()
$('#error').hide()

socket.on('connect', ()=>{
    console.log('connected!')
    socket.emit('start_game', (res)=>{
      if(res.code == 'success'){
        console.log('game has started! its your turn!')
        $('#buttons_and_slider').show()
      }
    })
})

socket.on('show_error', (res)=>{
    $('#error').show()
})

socket.on('change_chips', (res)=>{
    $("#player_chips").text(res.player_chips);
    $("#opponent_chips").text(res.opponent_chips);
})

socket.on('change_stats', (res)=>{
    $("#avg_wins").text(res.avg_wins)
})

socket.on('game_over', (res)=>{
    $('#waiting_for_player').hide()
    $('#waiting_for_bot').hide()
    $('#buttons_and_slider').hide()
    winner = res.winner
    socket.emit('player_received_end_game_msg')
    setTimeout(()=>{
        alert(winner.concat(' wins!'))
    }, 1000)
})

socket.on('new_turn', (res)=>{
    $('#waiting_for_player').hide()
    $('#waiting_for_bot').hide()
    $('#buttons_and_slider').hide()
    setTimeout(()=>{
        if(res.player == 'player'){
            $('#waiting_for_player').show()
            $('#buttons_and_slider').show()
        } else
        if(res.player == 'bot'){
            $('#waiting_for_bot').show()
        }
    }, 1000)
})

socket.on('change_cards', (res)=>{
    [b, p, opp] = [res.board_cards, res.player_cards, res.bot_cards]
    $('#p1').attr('src', 'pics/'.concat(p[0]).concat('.svg'));
    $('#p2').attr('src', 'pics/'.concat(p[1]).concat('.svg'));
    $('#b1').attr('src', 'pics/'.concat(b[0]).concat('.svg'));
    $('#b2').attr('src', 'pics/'.concat(b[1]).concat('.svg'));
    $('#b3').attr('src', 'pics/'.concat(b[2]).concat('.svg'));
    $('#b4').attr('src', 'pics/'.concat(b[3]).concat('.svg'));
    $('#b5').attr('src', 'pics/'.concat(b[4]).concat('.svg'));
    $('#opp1').attr('src', 'pics/'.concat(opp[0]).concat('.svg'));
    $('#opp2').attr('src', 'pics/'.concat(opp[1]).concat('.svg'));
})

$("#fold_button").click( ()=>{
    socket.emit('player_send_action', 'fold', -1, handle_action_response)
})

$("#call_button").click( ()=>{
    socket.emit('player_send_action', 'call', -1, handle_action_response)
})

$("#raise_button").click( ()=>{
    var mySlider = $("#slider").slider()
    var amount = mySlider.slider('getValue')
    amount -= amount%10
    socket.emit('player_send_action', 'raise', amount, handle_action_response)
})

$("#allin_button").click( ()=>{
    socket.emit('player_send_action', 'allin', -1, handle_action_response)
})

let handle_action_response = (res)=>{
    if(res.code == 'success'){
        console.log('Action:', res.action, res.amount)
    } else
    if(res.code == 'not_your_turn') {
        alert('Not Your Turn')
    } else
    if (res.code == 'not_allowed') {
        alert('Not Allowed Action')
    }
}
