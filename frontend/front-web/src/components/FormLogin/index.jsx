import React from "react";
import Input from "../../components/Inputs/index";
import Logo from "../../img/where2GoTxt.png";
import Btn from "../Btn";
import * as Styled from './style';

export default function(){
    return(
        <Styled.Content>
            <Styled.Img src={Logo}/>
            <div>
                <h2>Login</h2>
                <Input text="Usuário" type="text"/>
                <Input text="Senha" type="password"/>
            </div>

            
            <Btn text="Acessar" size="md" background="#5271ff"/>

        </Styled.Content>
    )
}