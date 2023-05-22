import React from "react";
import Input from "../../components/Inputs/index";
import Logo from "../../img/where2GoTxt.png";
import Btn from "../Btn";
import * as Styled from '../FormLogin/style';

export default function(){
    return(
        <Styled.Content>
            <Styled.Img src={Logo}/>
            <div>
            <h2>Cadastro</h2>
                <Input text="Nome usuário" type="text"/>
                <Input text="Nome" type="text"/>
                <Input text="Cpf" type="text"/>
                <Input text="Email" type="text"/>
                <Input text="Senha" type="password"/>
                <Input text="Confirme a senha" type="password"/>
                <Input text="Data Nascimento" type="date"/>
                <p>Ao se registrar, você aceita nossos 
                <Styled.A target="_blank" rel="noopener noreferrer" href="https://google.com.br/terms"> termos de uso </Styled.A>
                    e a nossa 
                    <Styled.A target="_blank" rel="noopener noreferrer" href="https://google.com.br/terms"> política de privacidade</Styled.A>
                .</p>
            </div>

            
            <Btn text="Acessar" size="md" background="#5271ff"/>

        </Styled.Content>
    )
}