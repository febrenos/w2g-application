import React, { useState, useEffect } from 'react'
import Btn from '../../components/Btn'
import Header from '../../components/Header/index'
import Test from '../../components/HeaderApresentation'
import Cellphone from '../../components/Cellphone'
import './style.css'
import TimeLine from '../../components/TimeLine/index'
import Loading from '../../components/Loading'
import SelectBtns from '../../components/SelectBtns'
import CardApresentation from '../../components/Cards/CardApresentation/index'
import Form from '../../components/FormLogin/index'
import FormRegister from '../../components/FormSignup'

export default function() {
    const [isOpen, setIsOpen] = useState(false);

    return(
        <>
        <main>
            <Header isOpen={isOpen} onClick={() => setIsOpen(!isOpen)}/> 
            <div className={isOpen ? 'contentClose' : 'contentOpen'}>
                <Btn text="BtnSmall" borderColor={"#0c899f"} size="sm"/>
                <Btn text="BtnMedium" borderColor={"#0c9f1e"} size="md"/>
                <Btn text="BtnLarge" size="lg" disabled/>
                <TimeLine steps={7} finished={6}/>
                {/* <SelectBtns>TEst</SelectBtns> */}
                <CardApresentation text="Get Started"/>
            </div>
        </main>
        </>
    )
}