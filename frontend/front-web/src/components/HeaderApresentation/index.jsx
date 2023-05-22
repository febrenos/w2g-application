import React, { useState } from "react";
import Logo from '../../img/w2gTxt.png';
import * as Styled from './style.jsx';
import Btn from '../Btn/index'


export default function(){
  return(
    <>
      <Styled.Menu>
        <Styled.Logo src={Logo}/>
        <p>Test</p>
        <p>Sobre nós</p>
        <Styled.Access>
          <p>Login</p>
          <Btn text="Cadastro"/>
        </Styled.Access>
      </Styled.Menu>
    </>
  )
}